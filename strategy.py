#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot levels from daily high-low-close + volume confirmation + volatility filter
# Long when price closes above Camarilla H4 level (resistance) with volume > 1.5x average and ATR(14) > 0.5*ATR(50)
# Short when price closes below Camarilla L4 level (support) with volume > 1.5x average and ATR(14) > 0.5*ATR(50)
# Exit when price touches Camarilla H3/L3 levels or volatility drops below threshold
# Position size: 0.25 (25% of capital)
# Uses daily Camarilla levels for structure, volume for confirmation, ATR ratio for volatility regime filter
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_camarilla_daily_vol_volat_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Camarilla levels: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    range_daily = high_daily - low_daily
    camarilla_h4 = close_daily + 1.1 * range_daily / 2
    camarilla_l4 = close_daily - 1.1 * range_daily / 2
    camarilla_h3 = close_daily + 1.1 * range_daily / 4
    camarilla_l3 = close_daily - 1.1 * range_daily / 4
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l3)
    
    # 4h volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: require ATR(14) > 0.5 * ATR(50) to avoid low-volatility chop
        vol_filter = atr14[i] > 0.5 * atr50[i]
        
        if position == 1:  # long position
            # Exit: price touches H3 level or volatility drops
            if close[i] <= h3_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches L3 level or volatility drops
            if close[i] >= l3_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and volatility filter
            # Long: price closes above H4 level, volume > 1.5x average, volatility sufficient
            if (close[i] > h4_aligned[i] and
                volume[i] > 1.5 * volume_ma[i] and
                vol_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price closes below L4 level, volume > 1.5x average, volatility sufficient
            elif (close[i] < l4_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i] and
                  vol_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals