#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Primary: 6h price breaking above/below Camarilla R4/S4 levels from prior 1d
# - Volume filter: 1d volume > 1.3x 20-period volume MA to confirm participation
# - Trend filter: 1w close > 50-period EMA (bullish bias) or < 50-period EMA (bearish bias)
# - Exit: Price returns to Camarilla pivot point (PP) or opposite S1/R1 level
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla adapts to volatility, volume confirms breakouts,
#   trend filter ensures alignment with higher timeframe momentum
# - Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe

name = "6h_1d_1w_camarilla_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels from prior 1d
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    # R1 = PP + (H - L) * 1.1/12
    # S1 = PP - (H - L) * 1.1/12
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s4_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    r1_1d = pp_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = pp_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 6h timeframe (using prior completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 1d volume confirmation: volume > 1.3x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1w trend filter: 50-period EMA
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period MA
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries at Camarilla R4/S4 breakouts
            # Long entry: price breaks above R4 + vol confirmation + 1w close > EMA50 (bullish bias)
            if close[i] > r4_aligned[i] and vol_confirm and close_1d[-1] > ema_50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S4 + vol confirmation + 1w close < EMA50 (bearish bias)
            elif close[i] < s4_aligned[i] and vol_confirm and close_1d[-1] < ema_50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at PP or opposite S1/R1
            # Exit: price reaches PP (mean reversion) or touches opposite S1/R1 (reversal)
            if position == 1:  # Long position
                if close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals