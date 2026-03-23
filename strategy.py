#!/usr/bin/env python3
"""
Experiment #012: 12h Primary + 1d/1w HTF — CRSI + Choppiness Regime + Donchian Breakout

Hypothesis: 12h timeframe with Connors RSI for entry timing + Choppiness regime filter
should generate 30-50 trades/year with higher win rate than simple trend following.
Key insight from #011 success: CRSI + Chop + Donchian achieved Sharpe=0.473 on 4h.
Adapting to 12h should reduce fee drag while maintaining edge.

Why this should work:
1. CRSI (Connors RSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Extreme values (<10 or >90) signal overbought/oversold with 75% win rate
   - Much faster than RSI(14), catches reversals earlier
2. Choppiness Index regime: CHOP > 55 = range (mean revert), CHOP < 45 = trend (breakout)
3. 1d HMA trend bias: Only trade with daily trend direction
4. Donchian(20) breakout confirmation in trending regime
5. 1w HMA macro filter: Avoid counter-trend trades against weekly trend
6. ATR(14) stoploss at 2.5x for risk management

Position size: 0.28 (discrete, within 0.20-0.35 range)
Target trades: 30-50/year on 12h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_donchian_regime_1d1w_v1"
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

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentage of past 100 days with lower returns
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very fast RSI
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * (1.0 - 1.0 / (streak[i] + 1))
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1.0 / (np.abs(streak[i]) + 1))
        else:
            streak_rsi[i] = 50.0
    
    # Smooth streak RSI with 2-period EMA
    streak_rsi_s = pd.Series(streak_rsi).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    # PercentRank(100) - percentage of past 100 returns lower than current
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    lookback = 100
    
    for i in range(lookback, n):
        current_return = returns[i]
        past_returns = returns[i-lookback:i]
        if len(past_returns) > 0:
            percent_rank[i] = 100.0 * np.sum(past_returns < current_return) / len(past_returns)
        else:
            percent_rank[i] = 50.0
    
    # CRSI calculation
    crsi = (rsi_3 + streak_rsi_s + percent_rank) / 3.0
    
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
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Also calculate price momentum
    roc_12 = pd.Series(close).pct_change(periods=12).values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donch_upper[i]) or atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-5] if i >= 5 else 0
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range regime
        is_trending = chop_value < 45.0  # Trend regime
        # Neutral zone: 45-55, use previous signal or stay flat
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        crsi_rising = crsi[i] > crsi[i-2] if i >= 2 else False
        crsi_falling = crsi[i] < crsi[i-2] if i >= 2 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donch_upper[i-1] if i >= 1 else False
        donchian_breakout_short = close[i] < donch_lower[i-1] if i >= 1 else False
        
        # === MOMENTUM FILTER ===
        momentum_positive = roc_12[i] > 0.0
        momentum_negative = roc_12[i] < 0.0
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + price near Donchian lower + daily trend neutral/bull
            if crsi_oversold and close[i] < donch_lower[i] * 1.02:
                if price_above_hma_1d or (not price_below_hma_1w):  # Not strongly bearish
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price near Donchian upper + daily trend neutral/bear
            elif crsi_overbought and close[i] > donch_upper[i] * 0.98:
                if price_below_hma_1d or (not price_above_hma_1w):  # Not strongly bullish
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Breakout with CRSI confirmation ---
        elif is_trending:
            # Long: Donchian breakout + CRSI rising + daily bullish + weekly confirms
            if donchian_breakout_long and crsi_rising:
                if price_above_hma_1d and momentum_positive:
                    if price_above_hma_1w or hma_1d_slope > 0:  # Weekly or daily momentum
                        new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + CRSI falling + daily bearish + weekly confirms
            elif donchian_breakout_short and crsi_falling:
                if price_below_hma_1d and momentum_negative:
                    if price_below_hma_1w or hma_1d_slope < 0:  # Weekly or daily momentum
                        new_signal = -POSITION_SIZE
        
        # --- NEUTRAL ZONE: CRSI extreme reversal ---
        else:
            # Very extreme CRSI can override regime
            if crsi_oversold and crsi_rising:
                if not price_below_hma_1w:  # Weekly not strongly bearish
                    new_signal = POSITION_SIZE
            elif crsi_overbought and crsi_falling:
                if not price_above_hma_1w:  # Weekly not strongly bullish
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
        
        # === EXIT ON REGIME/TREND CHANGE ===
        # Exit long if daily and weekly both turn bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and price_below_hma_1w:
                new_signal = 0.0
        
        # Exit short if daily and weekly both turn bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and price_above_hma_1w:
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