#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spike confirmation
# Camarilla R3/S3 levels act as strong intraday support/resistance. Breakouts with volume and 1d trend alignment
# capture sustained moves. Designed for low frequency (12-25 trades/year) to avoid fee drag in bear markets.
# Uses 6h timeframe with 1d HTF for structure and trend.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R3/S3 = close ± 1.1 * (high - low) / 2
    # Using previous day's values (shifted by 1)
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])  # shift(1)
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
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
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d Camarilla R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Camarilla S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d Camarilla S3
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d Camarilla R4 level (aggressive target)
            camarilla_r4 = prev_close[i] + 1.1 * (prev_high[i] - prev_low[i]) if i < len(prev_close) else curr_r3
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4) if i < len(prev_close) else curr_r3
            if curr_close >= camarilla_r4_aligned:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d Camarilla R3
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d Camarilla S4 level (aggressive target)
            camarilla_s4 = prev_close[i] - 1.1 * (prev_high[i] - prev_low[i]) if i < len(prev_close) else curr_s3
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4) if i < len(prev_close) else curr_s3
            if curr_close <= camarilla_s4_aligned:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals