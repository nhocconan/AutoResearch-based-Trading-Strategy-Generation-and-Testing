#!/usr/bin/env python3
"""
Experiment #009: 4h Primary + 1d HTF — Dual Regime with Connors RSI + Donchian

Hypothesis: 4h timeframe with daily trend bias captures medium-term swings while
avoiding fee drag from lower TF. Dual regime (mean revert in chop, trend in breakouts)
adapts to market conditions. Connors RSI provides high-probability mean reversion
entries (75% win rate in literature). Donchian breakout captures trend continuation.

Key components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(50)) / 3
   - Long when CRSI < 15 (oversold), Short when CRSI > 85 (overbought)
2. Choppiness Index: Regime detection (CHOP > 55 = range, CHOP < 40 = trend)
3. Donchian Channel(20): Breakout detection for trend regime
4. 1d HMA(21): Macro trend bias from daily timeframe
5. ATR(14) trailing stop: 2.5*ATR stoploss on all positions

Why this should work:
- 4h primary = 20-50 trades/year target (optimal fee/trade ratio)
- 1d HTF = strong trend filter, avoids counter-trend in strong moves
- Dual regime = adapts to market conditions (range vs trend)
- CRSI = proven mean reversion edge with tight entry thresholds
- Discrete position sizing (0.25) minimizes churn costs

Position size: 0.25 (discrete, conservative)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_chop_regime_1d_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=50):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(50)) / 3
    
    RSI_Streak: RSI of consecutive up/down periods
    PercentRank: Percentile rank of price change over lookback
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI of Streak (consecutive up/down periods)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # PercentRank: percentile of current price change over lookback
    price_change = close_s.diff()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = price_change.iloc[i-rank_period:i+1].values
        current = price_change.iloc[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    High CHOP (>61.8) = ranging, Low CHOP (<38.2) = trending
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (highest + lowest) / 2.0
    return highest, lowest, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=50)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Also calculate price position within Donchian channel
    donchian_range = donchian_high - donchian_low + 1e-10
    price_position = (close - donchian_low) / donchian_range  # 0=low, 1=high
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_high[i]) or atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Range regime
        is_trending = chop_value < 45.0  # Trend regime (hysteresis)
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong mean reversion long signal
        crsi_overbought = crsi[i] > 85.0  # Strong mean reversion short signal
        crsi_neutral_low = crsi[i] < 40.0
        crsi_neutral_high = crsi[i] > 60.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_upper = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_lower = close[i] < donchian_low[i-1]   # Break below previous low
        
        # === PRICE POSITION IN CHANNEL ===
        price_near_low = price_position[i] < 0.15  # Near Donchian low
        price_near_high = price_position[i] > 0.85  # Near Donchian high
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr_14[i] / (np.nanmean(atr_14[max(0,i-50):i]) + 1e-10)
        vol_elevated = atr_ratio > 1.2  # Volatility above recent average
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + price near Donchian low + daily bias helps
            if crsi_oversold or (price_near_low and crsi_neutral_low):
                if price_above_hma_1d or not price_below_hma_1d:  # Daily not strongly bearish
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price near Donchian high + daily bias helps
            elif crsi_overbought or (price_near_high and crsi_neutral_high):
                if price_below_hma_1d or not price_above_hma_1d:  # Daily not strongly bullish
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Donchian Breakout ---
        elif is_trending:
            # Long: Donchian breakout + CRSI not overbought + daily confirms
            if breakout_upper and crsi_neutral_low:
                if price_above_hma_1d:  # Daily trend confirmation
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + CRSI not oversold + daily confirms
            elif breakout_lower and crsi_neutral_high:
                if price_below_hma_1d:  # Daily trend confirmation
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: CRSI extreme reversal (always valid) ---
        if new_signal == 0.0:
            # Very extreme CRSI values override regime
            if crsi[i] < 10.0:  # Extremely oversold
                new_signal = POSITION_SIZE
            elif crsi[i] > 90.0:  # Extremely overbought
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if daily trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and chop_14[i] < 40.0:  # Trend regime + bearish daily
                new_signal = 0.0
        
        # Exit short if daily trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and chop_14[i] < 40.0:  # Trend regime + bullish daily
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