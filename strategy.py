#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot + Volume Spike + Daily Trend Filter
# Uses Camarilla pivot levels from daily timeframe for mean reversion entries
# Volume spike confirms institutional interest at pivot levels
# Daily EMA50 filter ensures we trade in direction of higher timeframe trend
# Works in bull/bear by fading extremes in ranging markets and following trend in trending markets
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily data
    # Based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * range_ * 1.1 / 2
    camarilla_h3 = prev_close + 1.1 * range_ * 1.1 / 4
    camarilla_h2 = prev_close + 1.1 * range_ * 1.1 / 6
    camarilla_l2 = prev_close - 1.1 * range_ * 1.1 / 6
    camarilla_l3 = prev_close - 1.1 * range_ * 1.1 / 4
    camarilla_l4 = prev_close - 1.1 * range_ * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x average volume (24-period ~ 12 days)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade long when price > daily EMA50, short when price < daily EMA50
        if price > ema_50_aligned[i]:
            # Uptrend bias - look for long entries at support
            if position == 0:
                # Long: price touches Camarilla L3 or L4 with volume spike
                if (price <= camarilla_l3_aligned[i] * 1.002 or price <= camarilla_l4_aligned[i] * 1.002) and vol > 1.5 * avg_vol[i]:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Hold long or exit at resistance
                if price >= camarilla_h3_aligned[i] * 0.998:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Flip from short to long if strong signal
                if (price <= camarilla_l3_aligned[i] * 1.002 or price <= camarilla_l4_aligned[i] * 1.002) and vol > 1.5 * avg_vol[i]:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
        else:
            # Downtrend bias - look for short entries at resistance
            if position == 0:
                # Short: price touches Camarilla H3 or H4 with volume spike
                if (price >= camarilla_h3_aligned[i] * 0.998 or price >= camarilla_h4_aligned[i] * 0.998) and vol > 1.5 * avg_vol[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == -1:
                # Hold short or exit at support
                if price <= camarilla_l3_aligned[i] * 1.002:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            elif position == 1:
                # Flip from long to short if strong signal
                if (price >= camarilla_h3_aligned[i] * 0.998 or price >= camarilla_h4_aligned[i] * 0.998) and vol > 1.5 * avg_vol[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0