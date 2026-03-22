#!/usr/bin/env python3
"""
Experiment #318: 1d Connors RSI Mean Reversion with HTF Trend Filter

Hypothesis: After #312 failed with HMA+RSI pullback (Sharpe=-0.999), the entry 
conditions were too restrictive for 1d timeframe. Research shows Connors RSI 
(CRSI) has 75% win rate for mean reversion within trends.

Key differences from failed #312:
1. CRSI (not simple RSI) - combines RSI(3) + Streak RSI + PercentRank
2. LOOSER entry thresholds (CRSI<15 instead of <10) for more trades
3. Single HTF (1w HMA) instead of dual - simpler, proven in #311
4. Asymmetric sizing: stronger positions when HTF confirms
5. ATR trailing stop at 2.5x (proven from #311 success)

Why this should work on 1d:
- 1d has fewer signals, so each must have high win rate
- CRSI captures oversold/overbought extremes better than RSI(14)
- Trend filter (SMA200 + 1w HMA) prevents counter-trend mean reversion
- Discrete sizing (0.20/0.30) minimizes fee churn on daily bars

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w HMA(21) via mtf_data helper (call ONCE before loop)
Position sizing: 0.20 base, 0.30 with HTF confirmation
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_meanrev_1w_hma_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_streak_rsi(close, period=2):
    """Calculate Streak RSI component of Connors RSI.
    
    Measures consecutive up/down days and converts to RSI-like scale.
    """
    n = len(close)
    streak_rsi = np.zeros(n)
    
    # Calculate streaks (consecutive up/down days)
    streaks = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streaks[i-1] > 0:
                streaks[i] = streaks[i-1] + 1
            else:
                streaks[i] = 1
        elif close[i] < close[i-1]:
            if streaks[i-1] < 0:
                streaks[i] = streaks[i-1] - 1
            else:
                streaks[i] = -1
        else:
            streaks[i] = 0
    
    # Convert streaks to RSI-like scale (0-100)
    # Positive streaks = bullish, negative = bearish
    for i in range(period, n):
        lookback = streaks[max(0, i-period+1):i+1]
        up_streaks = np.sum(lookback > 0)
        down_streaks = np.sum(lookback < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100 * up_streaks / total
        else:
            streak_rsi[i] = 50
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank component of Connors RSI.
    
    Measures where current price change ranks within recent history.
    """
    n = len(close)
    pct_rank = np.zeros(n)
    
    # Calculate daily returns
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100
    
    for i in range(period, n):
        lookback = returns[max(0, i-period+1):i+1]
        current = returns[i]
        # Count how many values in lookback are less than current
        rank = np.sum(lookback[:-1] < current)  # exclude current from comparison
        total = len(lookback) - 1
        if total > 0:
            pct_rank[i] = 100 * rank / total
        else:
            pct_rank[i] = 50
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI).
    
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Values range 0-100. <10 = extremely oversold, >90 = extremely overbought.
    """
    rsi_fast = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    pct_rank = calculate_percent_rank(close, rank_period)
    
    crsi = (rsi_fast + streak_rsi + pct_rank) / 3
    return crsi

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # CRSI thresholds (LOOSE for trade generation - Rule 9)
    CRSI_OVERSOLD = 15  # <15 = long signal (looser than standard 10)
    CRSI_OVERBOUGHT = 85  # >85 = short signal (looser than standard 90)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Start after SMA200 warmup + CRSI warmup
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1w HMA = meta-trend direction
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND FILTER ===
        # SMA200 = long-term trend direction
        bull_trend = close[i] > sma_200[i]
        bear_trend = close[i] < sma_200[i]
        
        # === CRSI MEAN REVERSION SIGNALS ===
        # Long: CRSI extremely oversold + in uptrend
        oversold = crsi[i] < CRSI_OVERSOLD
        # Short: CRSI extremely overbought + in downtrend
        overbought = crsi[i] > CRSI_OVERBOUGHT
        
        # === ENTRY CONDITIONS (LOOSE for >=10 trades) ===
        new_signal = 0.0
        position_size = SIZE_BASE
        
        # LONG: CRSI oversold + price > SMA200 (trend filter)
        # 1w HMA confirmation boosts size but not required for entry
        long_conditions = oversold and bull_trend
        
        # SHORT: CRSI overbought + price < SMA200 (trend filter)
        short_conditions = overbought and bear_trend
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            # Boost size if 1w HMA also confirms bull trend
            if bull_trend_1w:
                position_size = SIZE_STRONG
            new_signal = position_size
        
        if short_conditions:
            # Boost size if 1w HMA also confirms bear trend
            if bear_trend_1w:
                position_size = SIZE_STRONG
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price falls below SMA200
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            # Exit short if price rises above SMA200
            if position_side < 0 and bull_trend:
                new_signal = 0.0
        
        # === CRSI REVERSAL EXIT ===
        # Exit long if CRSI becomes overbought (mean reversion complete)
        if in_position and new_signal != 0.0:
            if position_side > 0 and overbought:
                new_signal = 0.0
            # Exit short if CRSI becomes oversold
            if position_side < 0 and oversold:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals