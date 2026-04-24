#!/usr/bin/env python3
"""
Hypothesis: 4h CRSI(2,14,100) with 1d Supertrend trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d Supertrend (ATR=10, mult=3.0) for trend filter (defines bull/bear regime).
- Entry: Long when CRSI < 15 in bull regime with volume > 1.3 * 4h volume MA(20);
         Short when CRSI > 85 in bear regime with volume > 1.3 * 4h volume MA(20).
- Exit: ATR trailing stop (2.5 * ATR(14)) or opposite CRSI extreme (CRSI>70 for long exit, CRSI<30 for short exit).
- Signal size: 0.25 discrete to balance capture and fee control.
- CRSI captures short-term mean reversion within trend; Supertrend filters regime; volume confirms conviction.
- Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for CRSI calculation and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(10) for Supertrend
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d Supertrend
    hl2 = (df_1d['high'] + df_1d['low']) / 2
    upper_band = hl2 + (3.0 * atr_1d)
    lower_band = hl2 - (3.0 * atr_1d)
    
    supertrend = np.zeros(len(df_1d))
    direction = np.ones(len(df_1d))  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(df_1d)):
        close_prev = df_1d['close'].iloc[i-1]
        supertrend_prev = supertrend[i-1]
        direction_prev = direction[i-1]
        
        if direction_prev == 1:
            supertrend[i] = max(lower_band[i], supertrend_prev) if close_prev > supertrend_prev else lower_band[i]
            direction[i] = -1 if close[i] < supertrend[i] else 1
        else:
            supertrend[i] = min(upper_band[i], supertrend_prev) if close_prev < supertrend_prev else upper_band[i]
            direction[i] = 1 if close[i] > supertrend[i] else -1
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate 4h RSI(2) for CRSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2 = rsi_2.fillna(50).values
    
    # Calculate 4h RSI(14) for CRSI streak component
    delta14 = pd.Series(close).diff()
    gain14 = delta14.clip(lower=0)
    loss14 = -delta14.clip(upper=0)
    avg_gain14 = gain14.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss14 = loss14.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs14 = avg_gain14 / avg_loss14.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs14))
    rsi_14 = rsi_14.fillna(50).values
    
    # Calculate 4h Percent Rank of RSI(14) over 100 periods
    rsi_14_series = pd.Series(rsi_14)
    percent_rank = rsi_14_series.rolling(window=100, min_periods=100).apply(
        lambda x: np.sum(x <= x.iloc[-1]) / len(x) * 100 if len(x) > 0 else 50, raw=False
    ).fillna(50).values
    
    # Calculate CRSI: (RSI(2) + RSI_Streak + PercentRank(100)) / 3
    # Simplified: Use RSI(2) as proxy for streak component in fast calculation
    crsi = (rsi_2 + 50 + percent_rank) / 3  # Approximation: streak ~50 when undefined
    # Better approximation: use RSI(14) for middle component when streak not available
    crsi = (rsi_2 + rsi_14 + percent_rank) / 3
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(30, 50, 20, 14, 100)  # Supertrend needs 30, RSI needs 50, volume MA needs 20, ATR needs 14, percent rank needs 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(crsi[i]) or np.isnan(vol_ma_4h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.3x threshold (balanced to reduce trades)
        vol_confirm = curr_volume > 1.3 * vol_ma_4h_aligned[i]
        
        # Trend filter: Supertrend direction
        bull_regime = direction_aligned[i] == 1
        bear_regime = direction_aligned[i] == -1
        
        if position == 0:
            # Check for entry signals
            # Long: CRSI < 15 (oversold) in bull regime with volume confirmation
            if crsi[i] < 15.0 and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: CRSI > 85 (overbought) in bear regime with volume confirmation
            elif crsi[i] > 85.0 and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or CRSI > 70 (exiting overbought)
            if curr_low <= highest_since_entry - 2.5 * atr[i] or crsi[i] > 70.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or CRSI < 30 (exiting oversold)
            if curr_high >= lowest_since_entry + 2.5 * atr[i] or crsi[i] < 30.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CRSI2_14_100_1dSupertrend_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0