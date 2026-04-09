#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal + 1d volume confirmation + 1w trend filter
# - Primary signal: Williams %R(14) crosses above/below oversold/overbought levels on 6h
# - Volume confirmation: 1d volume > 1.3x 20-period average (avoid low-volume false signals)
# - Trend filter: 1w EMA(50) slope determines bias - only long when EMA rising, short when falling
# - Works in bull/bear: In strong trends, Williams %R gives precise pullback entries
# - In ranging markets, mean reversion at extremes still works with volume confirmation
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) per 6h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "6h_1d_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.3 * avg_volume_20)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    # Pre-compute 1w EMA(50) slope for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope = np.diff(ema_50, prepend=ema_50[0])  # positive = rising trend
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope, additional_delay_bars=1)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, ((highest_high - close_6h) / hh_ll) * -100, -50)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_6h[0] = tr1[0]
    atr_14 = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_confirm_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (mean reversion) OR stoploss hit
            if williams_r[i] < -50 or close_6h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (mean reversion) OR stoploss hit
            if williams_r[i] > -50 or close_6h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R reversals with volume confirmation and trend filter
            # Long: Williams %R crosses above -80 from below (oversold bounce)
            # Short: Williams %R crosses below -20 from above (overbought rejection)
            if volume_confirm_aligned[i]:
                # Long condition: oversold bounce in rising trend OR any trend if strongly oversold
                if (williams_r[i-1] <= -80 and williams_r[i] > -80 and 
                    (ema_slope_aligned[i] > 0 or williams_r[i] < -90)):
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short condition: overbought rejection in falling trend OR any trend if strongly overbought
                elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and 
                      (ema_slope_aligned[i] < 0 or williams_r[i] > -10)):
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals