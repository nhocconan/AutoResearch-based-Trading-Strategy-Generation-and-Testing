#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels with 1w trend filter and volume spike
# - Primary signal: Price touches Camarilla H3 (resistance) or L3 (support) levels from prior 1d
# - HTF filter: 1w EMA(21) trend direction (above/below EMA)
# - Volume confirmation: 1d volume > 1.8x 20-period average volume
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 1.5x ATR(14) on 1d
# - Target: 7-25 trades/year (30-100 total over 4 years) per 1d strategy guidelines
# - Works in bull/bear: Pivot reversals capture retracements; trend filter avoids counter-trend trades

name = "1d_1w_camarilla_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Pre-compute 1d Camarilla pivot levels (based on prior day)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Calculate prior day's OHLC for Camarilla
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_high[0] = high_1d[0]  # first bar uses current
    prior_low[0] = low_1d[0]
    prior_close[0] = close_1d[0]
    
    pivot = (prior_high + prior_low + prior_close) / 3
    range_hl = prior_high - prior_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4)  # Resistance
    l3 = pivot - (range_hl * 1.1 / 4)  # Support
    h4 = pivot + (range_hl * 1.1 / 2)  # Strong resistance
    l4 = pivot - (range_hl * 1.1 / 2)  # Strong support
    
    # Pre-compute 1d volume spike filter
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.8 * avg_volume_20)
    
    # Pre-compute 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches L4 (strong support) OR stoploss hit
            if low_1d[i] <= l4[i] or low_1d[i] <= entry_price - 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches H4 (strong resistance) OR stoploss hit
            if high_1d[i] >= h4[i] or high_1d[i] >= entry_price + 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touches with volume and trend filters
            if vol_spike[i]:
                # Long: price touches L3 (support) in uptrend
                if low_1d[i] <= l3[i] and close_1d[i] > ema_21_aligned[i]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.25
                # Short: price touches H3 (resistance) in downtrend
                elif high_1d[i] >= h3[i] and close_1d[i] < ema_21_aligned[i]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.25
    
    return signals