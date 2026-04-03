#!/usr/bin/env python3
"""
Experiment #367: 6h Weekly Camarilla + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Weekly Camarilla pivot levels (calculated from prior week's OHLC) provide 
intraday support/resistance that respects institutional order flow. Combining these 
with 1d volume spikes (>1.8x 20-period average) confirms participation, while 1w EMA50 
trend filter ensures we trade with the higher timeframe momentum. Mean reversion at 
H3/L3 levels in trend direction, with breakouts at H4/L4 on volume spikes. Targets 
20-30 trades/year on 6h timeframe (80-120 total over 4 years) for optimal fee 
efficiency and statistical validity.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_camarilla_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for EMA50 trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
        # Trend: price above EMA = uptrend, below = downtrend
        trend_aligned = ema_aligned  # Will compare with close later
    else:
        ema_aligned = np.full(n, close.mean())  # Default to mean price
        trend_aligned = np.full(n, close.mean())
    
    # === HTF: 1w data for weekly Camarilla (Call ONCE before loop) ===
    # Weekly Camarilla levels for each 6h bar
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)  # Weekly close for reference
    
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed weekly bar before current 6h bar
        prior_weekly_bars = df_1w[df_1w['open_time'] < current_time]
        if len(prior_weekly_bars) > 0:
            prev_week = prior_weekly_bars.iloc[-1]
            ph = prev_week['high']
            pl = prev_week['low']
            pc = prev_week['close']
            
            # Weekly Camarilla formulas
            camarilla_close[i] = pc
            camarilla_h4 = pc + (ph - pl) * 1.1 / 2
            camarilla_l4 = pc - (ph - pl) * 1.1 / 2
            camarilla_h3 = pc + (ph - pl) * 1.1 / 4
            camarilla_l3 = pc - (ph - pl) * 1.1 / 4
            camarilla_h2 = pc + (ph - pl) * 1.1 / 6
            camarilla_l2 = pc - (ph - pl) * 1.1 / 6
            camarilla_h1 = pc + (ph - pl) * 1.1 / 12
            camarilla_l1 = pc - (ph - pl) * 1.1 / 12
            
            camarilla_h4[i] = camarilla_h4
            camarilla_l4[i] = camarilla_l4
            camarilla_h3[i] = camarilla_h3
            camarilla_l3[i] = camarilla_l3
            camarilla_h2[i] = camarilla_h2
            camarilla_l2[i] = camarilla_l2
            camarilla_h1[i] = camarilla_h1
            camarilla_l1[i] = camarilla_l1
        else:
            # Not enough prior data
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h2[i] = np.nan
            camarilla_l2[i] = np.nan
            camarilla_h1[i] = np.nan
            camarilla_l1[i] = np.nan
            camarilla_close[i] = np.nan
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Use 1w EMA50 ---
        is_uptrend = close[i] > trend_aligned[i]
        is_downtrend = close[i] < trend_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at H4 (strong resistance) or L4 (strong support) 
                if close[i] >= camarilla_h4[i] or close[i] <= camarilla_l4[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at H4 (strong resistance) or L4 (strong support)
                if close[i] >= camarilla_h4[i] or close[i] <= camarilla_l4[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price at L3 (mean reversion in uptrend) OR break above H3 with volume
        long_condition = is_uptrend and (
            (close[i] <= camarilla_l3[i] * 1.002 and close[i] >= camarilla_l3[i] * 0.998) or  # L3 mean reversion
            (close[i] > camarilla_h3[i] and volume_spike)  # Breakout above H3 with volume
        )
        
        # Short: Price at H3 (mean reversion in downtrend) OR break below L3 with volume
        short_condition = is_downtrend and (
            (close[i] >= camarilla_h3[i] * 0.998 and close[i] <= camarilla_h3[i] * 1.002) or  # H3 mean reversion
            (close[i] < camarilla_l3[i] and volume_spike)  # Breakdown below L3 with volume
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals