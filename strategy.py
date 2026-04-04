#!/usr/bin/env python3
"""
Experiment #5251: 6h Ichimoku Cloud + 1d Trend Filter + Volume Spike
HYPOTHESIS: On 6h timeframe, Ichimoku signals (Tenkan/Kijun cross) aligned with 1d cloud bias (price above/below 1d Kumo) capture institutional momentum with volume confirmation (>1.5x average volume). The 1d Ichimoku cloud acts as a higher-timeframe trend filter, reducing false signals in ranging markets. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets (long when price > 1d cloud + TK cross up + volume) and bear markets (short when price < 1d cloud + TK cross down + volume). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5251_6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Ichimoku Cloud (Senkou Span A/B, Kumo) ===
    if len(df_1d) >= 52:
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
        period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
        tenkan_sen = (period9_high + period9_low) / 2.0
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
        period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
        kijun_sen = (period26_high + period26_low) / 2.0
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
        period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2.0).shift(26)
        
        # Kumo (Cloud) boundaries: Senkou Span A and B
        # For trend filter: price above/below the cloud
        kumohigh = np.where(senkou_span_a >= senkou_span_b, senkou_span_a, senkou_span_b)
        kumolow = np.where(senkou_span_a <= senkou_span_b, senkou_span_a, senkou_span_b)
        
        # Align to 6h timeframe with shift(1) for completed 1d bars only
        kumohigh_aligned = align_htf_to_ltf(prices, df_1d, kumohigh.fillna(method='ffill').values)
        kumolow_aligned = align_htf_to_ltf(prices, df_1d, kumolow.fillna(method='ffill').values)
    else:
        kumohigh_aligned = np.full(n, np.nan)
        kumolow_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Ichimoku TK Cross (Tenkan/Kijun) ===
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2.0
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2.0
    
    # TK Cross signals: Tenkan crossing above/below Kijun
    tk_cross_up = (tenkan_sen_6h > kijun_sen_6h) & (tenkan_sen_6h.shift(1) <= kijun_sen_6h.shift(1))
    tk_cross_down = (tenkan_sen_6h < kijun_sen_6h) & (tenkan_sen_6h.shift(1) >= kijun_sen_6h.shift(1))
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(26, 20, 14)  # TK Cross, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(kumohigh_aligned[i]) or np.isnan(kumolow_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # 1d Cloud bias: price > Kumohigh = bullish bias, price < Kumolow = bearish bias
        cloud_bullish = price > kumohigh_aligned[i]
        cloud_bearish = price < kumolow_aligned[i]
        
        # 6h TK Cross signals
        tk_up_signal = tk_cross_up.iloc[i] if hasattr(tk_cross_up, 'iloc') else tk_cross_up[i]
        tk_down_signal = tk_cross_down.iloc[i] if hasattr(tk_cross_down, 'iloc') else tk_cross_down[i]
        
        # Final entry conditions
        # Long: TK cross up + price > 1d cloud (bullish alignment) + volume
        # Short: TK cross down + price < 1d cloud (bearish alignment) + volume
        if tk_up_signal and cloud_bullish and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif tk_down_signal and cloud_bearish and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals