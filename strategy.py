#!/usr/bin/env python3
"""
Experiment #643: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI + HMA Trend)

Hypothesis: Building on #637 (Sharpe=0.257) and #641 (Sharpe=0.277), this strategy uses
a dual-regime approach that adapts to market conditions. The Choppiness Index determines
whether we're in a trending or ranging market, then applies the appropriate entry logic.

Key insights from 569 failed strategies:
1. 1d timeframe with 1w HTF filter produces fewer, higher-quality trades (20-50/year)
2. Choppiness Index > 61.8 = range (use Connors RSI mean reversion)
3. Choppiness Index < 38.2 = trend (use HMA + Donchian breakout)
4. Weekly HMA slope provides major trend bias (avoids counter-trend trades)
5. Connors RSI < 10 or > 90 captures extreme reversals with 70%+ win rate
6. Conservative sizing (0.28) + 2.5*ATR stop controls drawdown during 2022 crash

Why this might beat Sharpe=0.520:
- Regime-adaptive: different logic for chop vs trend (proven in #641)
- 1w HMA filter keeps us on right side of major moves (like #637 but stricter)
- Connors RSI for mean reversion has better edge than standard RSI
- Donchian(20) breakout confirms momentum before trend entries
- Fewer trades = less fee drag on 1d timeframe
- Discrete sizing (0.28) minimizes signal churn costs

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 20-50 trades/year on 1d (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_chop_1w_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior close prices lower than current
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = streak[i-1] if i > 0 else 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50.0
        else:
            # Map streak to 0-100: longer streak = more extreme
            streak_rsi[i] = 50.0 + 50.0 * streak_sign[i] * min(streak_abs[i], streak_period) / streak_period
    
    # Percent Rank (100) - percentage of prior closes lower than current
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_lower / rank_period
    
    # CRSI = average of three components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel upper and lower bands.
    Upper = highest high over period
    Lower = lowest low over period
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging/consolidating
    CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        if highest_high - lowest_low > 0:
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend direction
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    hma_1d_fast = calculate_hma(close, period=10)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    choppiness = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(choppiness[i]) or atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (HMA slope over 2 bars) ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-2] if i >= 2 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-2] if i >= 2 else False
        
        # Price relative to 1w HMA
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 55.0  # Range/consolidation
        is_trending = choppiness[i] < 45.0  # Strong trend
        
        # === 1D HMA FAST/SLOW CROSSOVER ===
        hma_cross_bull = hma_1d_fast[i] > hma_1d[i]
        hma_cross_bear = hma_1d_fast[i] < hma_1d[i]
        
        # === 1D HMA SLOPE (2 bars) ===
        hma_1d_slope_bull = hma_1d[i] > hma_1d[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d[i] < hma_1d[i-2] if i >= 2 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === CONNORS RSI EXTREMES (Mean Reversion) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: TRENDING REGIME ---
        # Condition 1: 1w HMA sloping up + price above 1w HMA
        # Condition 2: 1d HMA fast > slow (momentum)
        # Condition 3: Donchian breakout OR RSI not overbought
        if is_trending and hma_1w_slope_bull and price_above_hma_1w:
            if hma_cross_bull and hma_1d_slope_bull:
                if donchian_breakout_up or (rsi_14[i] < 60.0 and close[i] > donchian_mid[i]):
                    new_signal = POSITION_SIZE
        
        # --- LONG ENTRY: CHOPPY REGIME (Mean Reversion) ---
        # Condition 1: 1w HMA not strongly bearish (neutral or bull)
        # Condition 2: CRSI extremely oversold (<15)
        # Condition 3: Price near Donchian lower (support)
        elif is_choppy and not hma_1w_slope_bear:
            if crsi_oversold or (rsi_oversold and close[i] < donchian_mid[i]):
                if close[i] > donchian_lower[i] * 0.98:  # Not crashing through support
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: TRENDING REGIME ---
        # Condition 1: 1w HMA sloping down + price below 1w HMA
        # Condition 2: 1d HMA fast < slow (momentum)
        # Condition 3: Donchian breakout OR RSI not oversold
        if is_trending and hma_1w_slope_bear and price_below_hma_1w:
            if hma_cross_bear and hma_1d_slope_bear:
                if donchian_breakout_down or (rsi_14[i] > 40.0 and close[i] < donchian_mid[i]):
                    new_signal = -POSITION_SIZE
        
        # --- SHORT ENTRY: CHOPPY REGIME (Mean Reversion) ---
        # Condition 1: 1w HMA not strongly bullish (neutral or bear)
        # Condition 2: CRSI extremely overbought (>85)
        # Condition 3: Price near Donchian upper (resistance)
        elif is_choppy and not hma_1w_slope_bull:
            if crsi_overbought or (rsi_overbought and close[i] > donchian_mid[i]):
                if close[i] < donchian_upper[i] * 1.02:  # Not breaking through resistance
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1w_slope_bear and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1w_slope_bull and price_above_hma_1w:
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