#!/usr/bin/env python3
# 12h_1d_camarilla_volume_crsi_v1
# Strategy: 12h Camarilla pivot breakout with volume confirmation and CRSI mean reversion filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels on 1d provide strong support/resistance. Breakouts with volume confirmation
# indicate institutional interest. CRSI(3,2,100) < 15 filters for oversold conditions in uptrends and > 85 for
# overbought in downtrends, improving win rate by avoiding false breakouts. Works in both bull (breakouts) and
# bear (mean reversion at pivots) markets. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_crsi_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Close + (High-Low) * multiplier
    # Key levels: H3/L3, H4/L4
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # CRSI calculation: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    # RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    avg_loss = loss.ewm(alpha=1/3, adjust=False, min_periods=3).mean()
    rs = avg_gain / avg_loss
    rsi3 = 100 - (100 / (1 + rs))
    
    # Streak RSI (2-period)
    up_days = (delta > 0).astype(int)
    down_days = (delta < 0).astype(int)
    streak_up = up_days * (up_days.groupby((up_days == 0).cumsum()).cumcount() + 1)
    streak_down = down_days * (down_days.groupby((down_days == 0).cumsum()).cumcount() + 1)
    streak_rsi_raw = streak_up - streak_down
    streak_rsi = pd.Series(streak_rsi_raw).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    streak_rsi = 100 * (streak_rsi + 1) / 2  # Scale to 0-100
    
    # Percent Rank (100-period)
    rank = pd.Series(close).rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    
    crsi = (rsi3 + streak_rsi + rank) / 3
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after sufficient data
        # Skip if any required data is invalid
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or 
            np.isnan(crsi.iloc[i]) if hasattr(crsi, 'iloc') else np.isnan(crsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get current CRSI value
        crsi_val = crsi.iloc[i] if hasattr(crsi, 'iloc') else crsi[i]
        
        # Long signal: price breaks above H3/H4 with volume and CRSI not overbought
        if (close[i] > h3_12h[i] and vol_confirm[i] and crsi_val < 85 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L3/L4 with volume and CRSI not oversold
        elif (close[i] < l3_12h[i] and vol_confirm[i] and crsi_val > 15 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to median level or CRSI extreme
        elif position == 1 and (close[i] < (h3_12h[i] + l3_12h[i]) / 2 or crsi_val > 90):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > (h3_12h[i] + l3_12h[i]) / 2 or crsi_val < 10):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals