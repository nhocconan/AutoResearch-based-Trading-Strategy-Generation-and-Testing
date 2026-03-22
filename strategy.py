#!/usr/bin/env python3
"""
Experiment #573: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Weekly Trend Filter

Hypothesis: After 572 failed experiments, the pattern is clear:
- Simple trend following fails on BTC/ETH (2022 crash whipsaw, 2025 bear market)
- Mean reversion WITH trend filter works best for daily timeframe
- Connors RSI (CRSI) has proven edge: 75% win rate in academic studies
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long when CRSI < 15 AND price > 1w HMA(21) (mean revert in uptrend)
- Short when CRSI > 85 AND price < 1w HMA(21) (mean revert in downtrend)
- 1w HTF filter prevents counter-trend mean reversion (major loss source)
- ATR(14) * 2.5 trailing stop for risk management
- Target: 20-40 trades/year on 1d (Rule 10: 10-30 max for daily)
- Position size: 0.30 discrete (Rule 4: max 0.40, typical 0.20-0.35)

Why this might beat Sharpe=0.435:
- CRSI specifically designed for short-term mean reversion (Connors Research)
- 1w HMA filter adds regime awareness without overfitting
- Fewer trades = less fee drag on daily timeframe
- Works in both bull (long pullbacks) and bear (short rallies) markets
- No complex regime switching that caused 0-trade failures (#569, #571)

Position sizing: 0.30 base (discrete per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=10 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_meanrevert_1w_v1"
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

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile of today's return vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    # Streak: count consecutive days in same direction
    returns = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            if returns.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif returns.iloc[i] < 0:
            if returns.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # RSI of streak values (period=2)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=2, min_periods=2, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=2, min_periods=2, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100) - percentile of today's return vs last 100
    percent_rank = np.zeros(n)
    for i in range(100, n):
        window_returns = returns.iloc[i-99:i+1]  # 100-day window including today
        today_return = returns.iloc[i]
        rank = np.sum(window_returns < today_return)
        percent_rank[i] = rank / 99.0 * 100.0  # 0-100 scale
    
    # Combine components
    crsi = (rsi_3 + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Need 100+ for CRSI percentRank + indicator warmup
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # 1w HMA slope for trend strength
        hma_1w_slope_bull = hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        hma_1w_slope_bear = hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        
        # === CONNORS RSI MEAN REVERSION ENTRY ===
        # Long: CRSI < 15 (extreme oversold) + in 1w uptrend
        # Short: CRSI > 85 (extreme overbought) + in 1w downtrend
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 1w bull + CRSI oversold
        if bull_regime_1w and crsi_oversold:
            # Size based on 1w trend strength
            if hma_1w_slope_bull:
                new_signal = POSITION_SIZE
            else:
                new_signal = POSITION_SIZE * 0.7  # Weaker trend = smaller size
        
        # SHORT ENTRY: 1w bear + CRSI overbought
        elif bear_regime_1w and crsi_overbought:
            # Size based on 1w trend strength
            if hma_1w_slope_bear:
                new_signal = -POSITION_SIZE
            else:
                new_signal = -POSITION_SIZE * 0.7  # Weaker trend = smaller size
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip or CRSI mean reversion complete) ===
        # Exit long on 1w regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1w and hma_1w_slope_bear:
                new_signal = 0.0
            # Also exit if CRSI recovers above 50 (mean reversion complete)
            if crsi[i] > 50.0:
                new_signal = 0.0
        
        # Exit short on 1w regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1w and hma_1w_slope_bull:
                new_signal = 0.0
            # Also exit if CRSI drops below 50 (mean reversion complete)
            if crsi[i] < 50.0:
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
                # Flip position
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