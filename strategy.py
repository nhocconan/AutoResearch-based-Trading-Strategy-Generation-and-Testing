#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability reversal/breakout zones. R3/S3 are strong resistance/support.
# 1d EMA(34) ensures trades align with daily trend, reducing false signals in ranging markets.
# Volume spike (>2.0x 20-bar average) confirms breakout momentum.
# Designed for low trade frequency (~20-50/year on 4h) to minimize fee drag and improve bear market performance.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (using typical price)
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    typical_price_s = pd.Series(typical_price)
    typical_price_ma_5 = typical_price_s.rolling(window=5, min_periods=5).mean().values
    typical_price_ma_5_aligned = align_htf_to_ltf(prices, df_1d, typical_price_ma_5)
    
    # Calculate Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    camarilla_range = df_1d['high'].values - df_1d['low'].values
    camarilla_r3 = df_1d['close'].values + 1.1 * camarilla_range
    camarilla_s3 = df_1d['close'].values - 1.1 * camarilla_range
    
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
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Camarilla R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Camarilla S3 (reversal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla R4 (strong resistance)
            camarilla_r4 = df_1d['close'].values[(i//24) if (i//24) < len(df_1d) else -1] + 1.5 * camarilla_range[(i//24) if (i//24) < len(df_1d) else -1]
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
            if curr_close >= camarilla_r4_aligned[i]:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Camarilla R3 (reversal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla S4 (strong support)
            camarilla_s4 = df_1d['close'].values[(i//24) if (i//24) < len(df_1d) else -1] - 1.5 * camarilla_range[(i//24) if (i//24) < len(df_1d) else -1]
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
            if curr_close <= camarilla_s4_aligned[i]:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals