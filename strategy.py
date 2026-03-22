#!/usr/bin/env python3
"""
Experiment #063: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: Daily timeframe with weekly trend filter can capture major moves while
avoiding whipsaw. Key innovations:

1. Connors RSI (CRSI) for entry timing — combines RSI(3) + Streak RSI(2) + PercentRank
   More responsive than standard RSI, catches short-term extremes better
2. Choppiness Index (CHOP) for regime detection — switches between:
   - CHOP > 55: Range/mean-reversion mode (fade extremes)
   - CHOP < 45: Trend mode (follow 1w HMA direction)
3. 1w HMA(21) for major trend bias — only trade in direction of weekly trend
4. ATR(14) stoploss at 2.5x — wider for daily timeframe volatility
5. Position size: 0.30 discrete — balanced for 20-50 trades/year target

Why this should work:
- 1d naturally limits trades to 30-60/year (within target)
- Connors RSI has 75% win rate in backtests (literature)
- Choppiness filter prevents trend strategies in ranges (major failure mode)
- 1w HMA prevents counter-trend trades in strong weekly trends
- Simpler than previous multi-regime attempts = more reliable trade generation

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_chop_regime_1w_hma_v1"
timeframe = "1d"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    Calculate RSI of streak (consecutive up/down days).
    Streak = number of consecutive days price moved in same direction.
    """
    n = len(close)
    streak = np.zeros(n)
    streak_direction = np.zeros(n)  # +1 for up streak, -1 for down streak
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak_direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
            streak_direction[i] = 1
        elif close[i] < close[i-1]:
            if streak_direction[i-1] == -1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
            streak_direction[i] = -1
        else:
            streak[i] = streak[i-1]
            streak_direction[i] = streak_direction[i-1]
    
    # Convert streak to RSI-like value (0-100)
    # High streak = extreme = overbought/oversold
    max_streak = 10  # Normalize to max 10 days
    streak_rsi = np.clip(streak / max_streak * 100, 0, 100)
    
    # Apply RSI calculation to streak values
    streak_s = pd.Series(streak_rsi)
    delta = streak_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    streak_rsi_final = 100 - (100 / (1 + rs))
    streak_rsi_final = streak_rsi_final.fillna(50).values
    
    return streak_rsi_final

def calculate_percent_rank(close, period=100):
    """
    Calculate Percentile Rank of today's return over last period days.
    Returns 0-100 value.
    """
    n = len(close)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    percent_rank = np.zeros(n)
    for i in range(period, n):
        window = returns[i-period+1:i+1]
        current = returns[i]
        # Count how many values in window are <= current
        rank = np.sum(window <= current) / len(window) * 100
        percent_rank[i] = rank
    
    return percent_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50  # Default to neutral
    
    return choppiness

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    choppiness = calculate_choppiness_index(high, low, close, 14)
    
    # Additional trend filters
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # Price above weekly HMA = bullish bias (prefer longs)
        # Price below weekly HMA = bearish bias (prefer shorts)
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # Weekly HMA slope (simple calculation)
        hma_slope_bullish = hma_1w_21_aligned[i] > hma_1w_21_aligned[i-5] if i >= 5 else False
        hma_slope_bearish = hma_1w_21_aligned[i] < hma_1w_21_aligned[i-5] if i >= 5 else False
        
        # === CHOPPINNESS REGIME ===
        # CHOP > 55 = range/mean-reversion mode
        # CHOP < 45 = trending mode
        # 45-55 = transition (reduce size)
        choppy_regime = choppiness[i] > 55
        trend_regime = choppiness[i] < 45
        transition_regime = not choppy_regime and not trend_regime
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 15 = oversold (long opportunity)
        # CRSI > 85 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_neutral = crsi[i] >= 15 and crsi[i] <= 85
        
        # === 1D TREND ALIGNMENT ===
        hma_21_above_50 = hma_1d_21[i] > hma_1d_50[i]
        hma_21_below_50 = hma_1d_21[i] < hma_1d_50[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transition regime
        if transition_regime:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if price_above_1w_hma or hma_slope_bullish:
            # Trend regime: enter on CRSI pullback (not extreme)
            if trend_regime:
                if crsi[i] < 40 and crsi[i] > 20 and hma_21_above_50:
                    new_signal = current_size
            # Choppy regime: enter on CRSI extreme (mean reversion)
            elif choppy_regime:
                if crsi_oversold and hma_21_above_50:
                    new_signal = current_size
            # Transition: only strong signals
            elif transition_regime:
                if crsi[i] < 25 and hma_21_above_50 and price_above_1w_hma:
                    new_signal = current_size * 0.7
        
        # SHORT ENTRIES
        if price_below_1w_hma or hma_slope_bearish:
            # Trend regime: enter on CRSI pullback (not extreme)
            if trend_regime:
                if crsi[i] > 60 and crsi[i] < 80 and hma_21_below_50:
                    new_signal = -current_size
            # Choppy regime: enter on CRSI extreme (mean reversion)
            elif choppy_regime:
                if crsi_overbought and hma_21_below_50:
                    new_signal = -current_size
            # Transition: only strong signals
            elif transition_regime:
                if crsi[i] > 75 and hma_21_below_50 and price_below_1w_hma:
                    new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 90 bars (~90 days on 1d), allow weaker entry
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if price_above_1w_hma and crsi[i] < 35 and hma_21_above_50:
                new_signal = current_size * 0.5
            elif price_below_1w_hma and crsi[i] > 65 and hma_21_below_50:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if weekly trend reverses bearish
            if position_side > 0 and price_below_1w_hma and hma_slope_bearish:
                trend_reversal = True
            # Exit short if weekly trend reverses bullish
            if position_side < 0 and price_above_1w_hma and hma_slope_bullish:
                trend_reversal = True
        
        # === CRSI EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long on CRSI overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Exit short on CRSI oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or crsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals