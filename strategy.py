#!/usr/bin/env python3
"""
Experiment #557: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: Based on experiment history, 1d strategies with regime detection work best:
- #546 (12h chop + connors + 1d) had Sharpe=-0.004 but +35.5% return (close to positive)
- Literature shows Connors RSI has 75% win rate on daily timeframes
- Choppiness Index effectively distinguishes trend vs range regimes
- 1w HTF provides major trend bias without over-filtering

Strategy logic:
1. 1w HMA(21) for major trend direction (HTF bias)
2. Choppiness Index(14) for regime detection:
   - CHOP > 55 = range regime → use Connors RSI mean reversion
   - CHOP < 45 = trend regime → use HMA pullback entries
3. Connors RSI for mean reversion entries (proven on ETH daily)
4. ATR(14) 2.5x trailing stoploss
5. Position size: 0.28 discrete (balanced for 1d frequency)

Why this might beat Sharpe=0.435:
- Dual regime adapts to market conditions (trend vs range)
- Connors RSI specifically designed for daily mean reversion
- 1w HTF filter prevents major counter-trend losses
- Simpler than failed multi-filter strategies (#550, #552 had 0 trades)
- 1d timeframe = 20-50 trades/year target (optimal per Rule 10)

Position sizing: 0.28 base (discrete, max 0.40 per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_connors_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = (ATR(1, n) / (Highest High(n) - Lowest Low(n))) * 100
    - CHOP > 61.8 = ranging market (mean reversion)
    - CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    
    # ATR(1) = simple true range (no smoothing)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR(1) over n periods
    tr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    
    # Highest High and Lowest Low over n periods
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Choppiness Index
    price_range = highest_high - lowest_low
    chop = np.where(price_range > 0, (tr_sum / price_range) * 100.0, 50.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): 3-period RSI of close
    2. RSI_Streak(2): 2-period RSI of streak length
    3. PercentRank(100): percentile rank of today's return over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak Length
    # Streak = consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    
    for i in range(1, n):
        if delta.iloc[i] > 0:
            if i > 0 and delta.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta.iloc[i] < 0:
            if i > 0 and delta.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # RSI(2) of the streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak_vals = rsi_streak.values
    
    # Component 3: Percentile Rank of Returns (over last 100 days)
    returns = close_s.pct_change() * 100.0
    percent_rank = np.zeros(n)
    
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i].values
        current = returns.iloc[i]
        if len(window) > 0:
            percent_rank[i] = np.sum(window < current) / len(window) * 100.0
    
    # Combine components
    crsi = (rsi_3 + rsi_streak_vals + percent_rank) / 3.0
    
    return crsi

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
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # 1d HMA for trend entries
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # 1w HMA slope for trend strength
        hma_1w_slope_bull = hma_1w_21_aligned[i] > hma_1w_50_aligned[i]
        hma_1w_slope_bear = hma_1w_21_aligned[i] < hma_1w_50_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range regime (mean reversion)
        # CHOP < 45 = trend regime (trend follow)
        # 45-55 = neutral (no new entries, hold existing)
        range_regime = chop_14[i] > 55.0
        trend_regime = chop_14[i] < 45.0
        
        # === 1D HMA TREND ===
        bull_1d = close[i] > hma_1d_21[i]
        bear_1d = close[i] < hma_1d_21[i]
        hma_1d_slope_bull = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_slope_bear = hma_1d_21[i] < hma_1d_50[i]
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # RANGE REGIME: Connors RSI mean reversion
        if range_regime:
            # Long: CRSI < 15 (extreme oversold) + 1w bull bias
            if crsi[i] < 15.0 and bull_regime_1w:
                new_signal = POSITION_SIZE
            
            # Short: CRSI > 85 (extreme overbought) + 1w bear bias
            elif crsi[i] > 85.0 and bear_regime_1w:
                new_signal = -POSITION_SIZE
        
        # TREND REGIME: HMA pullback entries
        elif trend_regime:
            # Long: 1w bull + 1d bull + pullback to HMA21
            if bull_regime_1w and bull_1d and hma_1d_slope_bull:
                # Check if price pulled back to HMA21 (within 2%)
                pullback_long = (close[i] - hma_1d_21[i]) / hma_1d_21[i] < 0.02
                if pullback_long:
                    new_signal = POSITION_SIZE
            
            # Short: 1w bear + 1d bear + rally to HMA21
            elif bear_regime_1w and bear_1d and hma_1d_slope_bear:
                # Check if price rallied to HMA21 (within 2%)
                rally_short = (hma_1d_21[i] - close[i]) / hma_1d_21[i] < 0.02
                if rally_short:
                    new_signal = -POSITION_SIZE
        
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 1w regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1w and hma_1w_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1w regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1w and hma_1w_slope_bull:
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