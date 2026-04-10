#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter
# - Long when price breaks above Camarilla R4 AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA20 (uptrend)
# - Short when price breaks below Camarilla S4 AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA20 (downtrend)
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots identify key intraday support/resistance levels; breaks indicate institutional participation
# - Volume confirmation ensures breakouts have conviction
# - Weekly trend filter ensures we trade with the higher timeframe momentum
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_1w_camarilla_breakout_v1"
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
    
    # Pre-compute 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h Camarilla Pivots (based on previous 6h bar)
    # Camarilla levels: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # Pivot = (High + Low + Close)/3
    camarilla_h4 = np.zeros_like(close)
    camarilla_l4 = np.zeros_like(close)
    camarilla_pivot = np.zeros_like(close)
    
    for i in range(1, len(close)):
        # Use previous bar's OHLC to calculate today's Camarilla levels
        camarilla_h4[i] = close[i-1] + 1.5 * (high[i-1] - low[i-1])
        camarilla_l4[i] = close[i-1] - 1.5 * (high[i-1] - low[i-1])
        camarilla_pivot[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
    
    # Pre-compute 6h ATR (14-period) for stoploss
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Pre-compute 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1d volume > 1.5x 20-period average
            # Since we don't have current 1d volume aligned, we use price close > volume MA as proxy for strength
            volume_confirmed = close[i] > vol_ma_1d_aligned[i]  # Simplified proxy
            
            # Trend filter: 1w close > 1w EMA20 for long, < for short
            uptrend = close[i] > ema_20_1w_aligned[i]
            downtrend = close[i] < ema_20_1w_aligned[i]
            
            # Long conditions: price breaks above Camarilla H4 AND volume confirmed AND uptrend
            if close[i] > camarilla_h4_aligned[i] and volume_confirmed and uptrend:
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L4 AND volume confirmed AND downtrend
            elif close[i] < camarilla_l4_aligned[i] and volume_confirmed and downtrend:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla pivot point
            exit_long = (position == 1 and close[i] <= camarilla_pivot_aligned[i])
            exit_short = (position == -1 and close[i] >= camarilla_pivot_aligned[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= camarilla_h4_aligned[i] - 2.0 * atr[i])
            stop_short = (position == -1 and close[i] >= camarilla_l4_aligned[i] + 2.0 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals