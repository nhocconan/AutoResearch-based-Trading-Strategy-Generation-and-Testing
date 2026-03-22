#!/usr/bin/env python3
"""
Experiment #037: 1d Primary + 1w HTF — Dual Regime Adaptive Strategy

Hypothesis: Daily timeframe captures major moves while avoiding noise. 
Weekly trend filter provides major bias. Dual regime adapts to market conditions:
- Range market (CHOP>50): Mean-revert using Connors RSI extremes
- Trend market (CHOP<50): Trend-follow using Donchian breakouts

Key changes from failed strategies:
1. LOOSENED entry thresholds (CRSI 25/75 instead of 15/85) to ensure trades
2. Removed session filter (not relevant for 1d)
3. Simplified regime logic (single CHOP threshold, not multiple)
4. Always allow entries when HTF trend aligns (don't require both regimes)

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (conservative for daily bars)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-80/year per symbol (looser entries than previous attempts)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_connors_donchian_1w_v2"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Loosened thresholds for more trades: <25 oversold, >75 overbought
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.fillna(50).values
    
    # Component 3: Percent Rank
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= rank_period else 50
    )
    percent_rank = percent_rank.fillna(50).values
    
    # Connors RSI
    connors_rsi = (rsi_3 + streak_rsi + percent_rank) / 3
    return connors_rsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    
    CHOP > 50 = range/choppy (mean reversion)
    CHOP < 50 = trending (trend following)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    
    chop = 100 * (atr1_sum / atr_period) / hh_ll * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

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
    connors_rsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(connors_rsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 50 = range (mean reversion preferred)
        # CHOP < 50 = trend (trend following preferred)
        is_range = chop[i] > 50
        is_trend = chop[i] < 50
        
        # === CONNORS RSI EXTREMES (LOOSENED for more trades) ===
        # CRSI < 25 = oversold (long opportunity) - was 15
        # CRSI > 75 = overbought (short opportunity) - was 85
        crsi_oversold = connors_rsi[i] < 25
        crsi_overbought = connors_rsi[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        # Breakout above previous high = long signal
        # Breakout below previous low = short signal
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (LOOSENED - multiple paths to entry) ===
        new_signal = 0.0
        
        # PATH 1: Mean Reversion in Range Market
        # Only need: range regime + CRSI extreme + HTF trend alignment
        if is_range:
            if crsi_oversold and trend_1w_bullish:
                new_signal = BASE_SIZE
            elif crsi_overbought and trend_1w_bearish:
                new_signal = -BASE_SIZE
        
        # PATH 2: Trend Following in Trend Market
        # Only need: trend regime + Donchian breakout + HTF trend alignment
        if is_trend:
            if breakout_long and trend_1w_bullish:
                new_signal = BASE_SIZE
            elif breakout_short and trend_1w_bearish:
                new_signal = -BASE_SIZE
        
        # PATH 3: Strong HTF Trend Override (always allow if HTF trend is strong)
        # This ensures we get trades even if regime filters don't trigger
        if trend_1w_bullish and crsi_oversold:
            new_signal = max(new_signal, BASE_SIZE * 0.5)
        if trend_1w_bearish and crsi_overbought:
            new_signal = min(new_signal, -BASE_SIZE * 0.5)
        
        # PATH 4: Breakout regardless of regime if HTF confirms
        if trend_1w_bullish and breakout_long:
            new_signal = max(new_signal, BASE_SIZE)
        if trend_1w_bearish and breakout_short:
            new_signal = min(new_signal, -BASE_SIZE)
        
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
        
        # Apply stoploss
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals