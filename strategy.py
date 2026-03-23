#!/usr/bin/env python3
"""
Experiment #136: 12h Primary + 1d HTF — Dual Regime (Trend + Mean Reversion)

Hypothesis: Previous 12h strategies failed due to over-complex regime detection
(Choppiness Index caused 0 trades in #128, #130, #131, #132). This uses simpler
1d HMA slope for regime detection + Connors RSI for mean reversion entries.

REGIME DETECTION (1d HMA slope):
- Strong trend (|slope| > 1.0%): Follow Donchian breakouts in trend direction
- Weak/flat trend (|slope| <= 1.0%): Mean revert on RSI extremes with SMA200 filter

ENTRY LOGIC:
1) TREND MODE: 1d HMA slope > 1.0% + price > 1d HMA + Donchian(20) breakout → Long
               1d HMA slope < -1.0% + price < 1d HMA + Donchian(20) breakout → Short
2) MEAN REVERT MODE: 1d HMA flat + RSI(3) < 15 + price > SMA200 → Long
                     1d HMA flat + RSI(3) > 85 + price < SMA200 → Short

EXIT LOGIC:
- ATR(14) trailing stop at 2.5x
- Opposite Donchian break
- RSI extreme take-profit (RSI > 80 for longs, < 20 for shorts)

Why this should work:
- Simpler regime detection = more trades (avoids 0-trade failure)
- Connors RSI proven on ETH (Sharpe +0.923 in research notes)
- 12h timeframe naturally limits trades to 20-50/year
- Dual mode captures both trending and ranging markets

Position size: 0.25 base, 0.30 with volume confirmation
Stoploss: 2.5*ATR trailing
Target: Sharpe > 0.5 on ALL symbols, 25-40 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_rsi(close, period=3):
    """Calculate RSI for Connors RSI (short period for mean reversion)."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_rsi_streak(close, period=2):
    """Calculate RSI Streak component for Connors RSI."""
    # Count consecutive up/down days
    direction = np.sign(np.diff(close))
    direction = np.insert(direction, 0, 0)
    
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if direction[i] > 0:
            if direction[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif direction[i] < 0:
            if direction[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert to RSI-like scale (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(period, len(close)):
        up_streaks = np.sum(streak[i-period+1:i+1] > 0)
        streak_rsi[i] = (up_streaks / period) * 100.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank for Connors RSI."""
    pr = np.zeros(len(close))
    for i in range(period, len(close)):
        current_return = (close[i] - close[i-1]) / (close[i-1] + 1e-10)
        past_returns = (close[i-period+1:i] - close[i-period:i-1]) / (close[i-period:i-1] + 1e-10)
        pr[i] = (np.sum(past_returns < current_return) / period) * 100.0
    return pr

def calculate_connors_rsi(close):
    """Calculate Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3."""
    rsi_3 = calculate_rsi(close, period=3)
    rsi_streak = calculate_rsi_streak(close, period=2)
    pr_100 = calculate_percent_rank(close, period=100)
    crsi = (rsi_3 + rsi_streak + pr_100) / 3.0
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for regime detection
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope (regime indicator)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    crsi = calculate_connors_rsi(close)
    hma_12h_21 = calculate_hma(close, period=21)
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_12h_21[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === REGIME DETECTION (1d HMA slope) ===
        trend_mode = abs(hma_1d_slope[i]) > 1.0
        mean_revert_mode = abs(hma_1d_slope[i]) <= 1.0
        
        # === TREND DIRECTION ===
        bullish_regime = hma_1d_slope[i] > 0.5 and close[i] > hma_1d_aligned[i]
        bearish_regime = hma_1d_slope[i] < -0.5 and close[i] < hma_1d_aligned[i]
        neutral_regime = abs(hma_1d_slope[i]) <= 0.5
        
        # === 12h TREND FILTER ===
        hma_12h_bullish = close[i] > hma_12h_21[i]
        hma_12h_bearish = close[i] < hma_12h_21[i]
        
        # === DONCHIAN BREAKOUT ===
        prev_high = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_low = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        breakout_long = close[i] > prev_high
        breakout_short = close[i] < prev_low
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.3
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND MODE ENTRY ---
        if trend_mode:
            # Long: bullish regime + 12h bullish + Donchian breakout
            if bullish_regime and hma_12h_bullish and breakout_long:
                new_signal = POSITION_SIZE_BASE
                if volume_confirmed:
                    new_signal = POSITION_SIZE_MAX
            
            # Short: bearish regime + 12h bearish + Donchian breakout
            if bearish_regime and hma_12h_bearish and breakout_short:
                new_signal = -POSITION_SIZE_BASE
                if volume_confirmed:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- MEAN REVERSION MODE ENTRY ---
        if mean_revert_mode:
            # Long: CRSI oversold + price above SMA200 (long-term uptrend intact)
            if crsi_oversold and close[i] > sma_200[i]:
                new_signal = POSITION_SIZE_BASE
            
            # Short: CRSI overbought + price below SMA200 (long-term downtrend intact)
            if crsi_overbought and close[i] < sma_200[i]:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price above Donchian mid and not overbought
                if close[i] > donchian_mid[i] and crsi[i] < 85:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price below Donchian mid and not oversold
                if close[i] < donchian_mid[i] and crsi[i] > 15:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if bearish_regime or breakout_short:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if bullish_regime or breakout_long:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 85:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 15:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals