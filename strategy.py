#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla R3/S3 levels with 1d trend filter and volume confirmation
# In ranging markets, price tends to revert from Camarilla R3/S3 levels; in trending markets,
# breaks of R4/S4 with 1d EMA alignment and volume spike indicate continuation.
# Uses discrete position sizing (0.0, ±0.25) to limit fee drawdown and ensure sufficient trades.
# Designed for low trade frequency (~20-40/year) to thrive in both bull and bear regimes.

name = "6h_Camarilla_R3S3_R4S4_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar's range)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C, H, L are from the *completed* 12h bar
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate levels using the *previous* completed 12h bar (to avoid look-ahead)
    # For current 12h bar index j, use bar j-1's OHLC
    prev_close = np.roll(close_12h, 1)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close[0] = np.nan  # first bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    rang = prev_high - prev_low
    r4 = prev_close + rang * 1.1 / 2.0
    r3 = prev_close + rang * 1.1 / 4.0
    s3 = prev_close - rang * 1.1 / 4.0
    s4 = prev_close - rang * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss on 6h timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma_20 = np.mean(volume[:i])
        else:
            vol_ma_20 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_r4 = r4_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_s4 = s4_aligned[i]
        
        # Skip if any Camarilla level is NaN (first bar)
        if np.isnan(curr_r4) or np.isnan(curr_r3) or np.isnan(curr_s3) or np.isnan(curr_s4):
            continue
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish breakout: price breaks above R4 with 1d uptrend
                if curr_close > curr_r4 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below S4 with 1d downtrend
                elif curr_close < curr_s4 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                # Bullish mean reversion: price rejects from S3 with 1d uptrend
                elif curr_close < curr_s3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish mean reversion: price rejects from R3 with 1d downtrend
                elif curr_close > curr_r3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks S4 (invalidates bullish structure)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s4:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches R3 (mean reversion target) or R4 (breakout target)
            elif curr_close >= curr_r3:
                signals[i] = 0.0  # exit full position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks R4 (invalidates bearish structure)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r4:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches S3 (mean reversion target) or S4 (breakout target)
            elif curr_close <= curr_s3:
                signals[i] = 0.0  # exit full position
            else:
                signals[i] = -0.25
    
    return signals