#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H4/L4 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above H4 with volume spike AND 12h EMA50 uptrend
# Short when price breaks below L4 with volume spike AND 12h EMA50 downtrend
# Uses proven Camarilla pivot structure with tighter H4/L4 levels (stronger support/resistance) 
# to reduce false breakouts and overtrading. 12h EMA50 provides smoother trend filter than 1d.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag while capturing strong breakouts

name = "4h_Camarilla_H4L4_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4/L4 are stronger levels (closer to price)
    # H4 = close + (high - low) * 1.1/4
    # L4 = close - (high - low) * 1.1/4
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (range_1d * 1.1 / 4)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_h4 = h4_aligned[i]
        curr_l4 = l4_aligned[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 12h EMA50, bearish if price < 12h EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation to avoid false breakouts
            if curr_volume_confirm:
                # Bullish entry: price breaks above H4 with volume AND bullish regime
                if curr_high > curr_h4 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below L4 with volume AND bearish regime
                elif curr_low < curr_l4 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when price falls below L4 or regime changes
            if curr_low < curr_l4 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price rises above H4 or regime changes
            if curr_high > curr_h4 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals