#!/usr/bin/env python3
"""
Experiment #007: 6h Weekly EMA200 Trend + RSI Momentum

HYPOTHESIS: BTC crashed 77% in 2022. Shorting in that environment destroyed gains.
This strategy is LONG-ONLY with weekly EMA200 as structural trend filter.
RSI(14) provides mean-reversion entries within the uptrend (buy oversold, sell overbought).
Volume confirmation ensures institutional participation.

WHY IT WORKS IN BULL AND BEAR:
- Bull (2021, 2024+): price above weekly EMA200 → RSI oversold = buy the dip
- Bear (2022): price below weekly EMA200 → NO ENTRIES → avoids the crash
- Range: price above weekly EMA200 but in range → RSI extremes still work

TARGET: 50-150 total trades over 4 years = 12-37/year.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_ema200_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(prices, period=14):
    """RSI indicator"""
    n = len(prices)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA200 for structural trend (bull/bear filter)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 6h indicators ===
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # RSI extremes for entry
    RSI_OVERSOLD = 35      # Not -80 for faster entries
    RSI_OVERBOUND = 45     # Exit when RSI recovers to here
    RSI_EXIT = 65          # Take profit when overbought
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    
    warmup = 500  # Need enough for 200-period weekly EMA alignment buffer
    
    for i in range(warmup, n):
        # Skip if RSI not ready
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if weekly EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND FILTER: Weekly EMA200 ===
        bull_market = close[i] > ema_1w_aligned[i]
        
        # Volume confirmation (required for entry)
        vol_confirm = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI oversold + bull market + volume ===
            if bull_market and rsi_14[i] < RSI_OVERSOLD and vol_confirm:
                desired_signal = SIZE
        
        if in_position and position_side > 0:
            # === HOLD/EXIT: Long position management ===
            
            # Take profit when RSI overbought
            if rsi_14[i] > RSI_EXIT:
                desired_signal = 0.0
            else:
                # Exit when RSI recovers to neutral-overbought
                if rsi_14[i] > RSI_OVERBOUND:
                    desired_signal = SIZE / 2  # Half position (take partial profit)
                else:
                    desired_signal = SIZE  # Hold full position
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or add/remove
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals