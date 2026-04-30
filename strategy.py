#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Camarilla pivot levels (R3/S3) act as strong support/resistance. Breakouts above R3 or below S3 with volume confirmation
# capture momentum moves. The 1d EMA(34) ensures trades align with the higher-timeframe trend, reducing false breakouts.
# Designed for low trade frequency (~19-50/year on 4h) to minimize fee drag and improve performance in both bull and bear markets.

name = "4h_WeeklyCamarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on prior week's OHLC)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = (H+L+CLOSE)/3 (typical price)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    hl_range = df_1w['high'] - df_1w['low']
    camarilla_r3 = typical_price + hl_range * 1.1 / 4
    camarilla_s3 = typical_price - hl_range * 1.1 / 4
    
    # Align weekly Camarilla levels to 4h timeframe (wait for weekly bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3.values)
    
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
    
    start_idx = 35  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
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
                # Bullish entry: price breaks above weekly R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below weekly S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks weekly S3 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches weekly R4 (mean reversion tendency)
            # R4 = C + (H-L)*1.1/2 = R3 + (H-L)*1.1/4
            hl_range_1w = (df_1w['high'].iloc[-1] - df_1w['low'].iloc[-1]) if len(df_1w) > 0 else 0
            typical_price_1w = (df_1w['high'].iloc[-1] + df_1w['low'].iloc[-1] + df_1w['close'].iloc[-1]) / 3 if len(df_1w) > 0 else 0
            camarilla_r4 = typical_price_1w + hl_range_1w * 1.1 / 2
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(df_1w['close'].values, camarilla_r4))[i] if len(df_1w) > 0 else curr_r3
            if curr_close >= camarilla_r4_aligned:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks weekly R3 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches weekly S4 (mean reversion tendency)
            # S4 = C - (H-L)*1.1/2 = S3 - (H-L)*1.1/4
            hl_range_1w = (df_1w['high'].iloc[-1] - df_1w['low'].iloc[-1]) if len(df_1w) > 0 else 0
            typical_price_1w = (df_1w['high'].iloc[-1] + df_1w['low'].iloc[-1] + df_1w['close'].iloc[-1]) / 3 if len(df_1w) > 0 else 0
            camarilla_s4 = typical_price_1w - hl_range_1w * 1.1 / 2
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(df_1w['close'].values, camarilla_s4))[i] if len(df_1w) > 0 else curr_s3
            if curr_close <= camarilla_s4_aligned:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals