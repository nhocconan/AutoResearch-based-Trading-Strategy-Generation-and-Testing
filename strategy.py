#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above R3 AND price > 1d EMA34 (bullish regime) AND volume > 2x 20-period MA
# Short when price breaks below S3 AND price < 1d EMA34 (bearish regime) AND volume > 2x 20-period MA
# Exit when price returns to Camarilla H3/L3 levels or regime changes
# Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Works in bull/bear markets by following 1d trend while capturing 4h momentum

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    #          H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    #          H2 = close + 1.1*(high-low)/6, L2 = close - 1.1*(high-low)/6
    #          H1 = close + 1.1*(high-low)/12, L1 = close - 1.1*(high-low)/12
    #          Pivot = (high + low + close)/3
    # We focus on R3/H3 and S3/L3 for breakouts
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.0x 20-period average (20*4h = ~3.3 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_open = open_[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 (H3) AND bullish regime
                if curr_high > curr_h3 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 (L3) AND bearish regime
                elif curr_low < curr_l3 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to H3 level OR regime changes to bearish
            if curr_close <= curr_h3 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to L3 level OR regime changes to bullish
            if curr_close >= curr_l3 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals