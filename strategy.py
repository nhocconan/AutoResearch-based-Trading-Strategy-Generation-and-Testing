#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Camarilla R3/S3 levels from daily chart act as strong support/resistance. Breakouts above R3 or below S3 with volume confirmation
# capture momentum moves. The 1d EMA(34) ensures trades align with the daily trend, reducing false breakouts.
# Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag and improve test generalization.
# Works in both bull and bear markets by following the higher-timeframe trend.

name = "12h_1dCamarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on prior 1d bar's OHLC)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = typical_price + hl_range * 1.1 / 4
    camarilla_s3 = typical_price - hl_range * 1.1 / 4
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
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
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d S3 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d R4 (mean reversion tendency)
            hl_range_1d = df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1] if len(df_1d) > 0 else 0
            typical_price_1d = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else 0
            camarilla_r4 = typical_price_1d + hl_range_1d * 1.1 / 2
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(df_1d['close'].values, camarilla_r4))[i] if len(df_1d) > 0 else curr_r3
            if curr_close >= camarilla_r4_aligned:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d R3 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d S4 (mean reversion tendency)
            hl_range_1d = df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1] if len(df_1d) > 0 else 0
            typical_price_1d = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else 0
            camarilla_s4 = typical_price_1d - hl_range_1d * 1.1 / 2
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(df_1d['close'].values, camarilla_s4))[i] if len(df_1d) > 0 else curr_s3
            if curr_close <= camarilla_s4_aligned:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals