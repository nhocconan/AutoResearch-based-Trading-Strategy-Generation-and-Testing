#!/usr/bin/env python3
"""
Experiment #556: 12h Primary + 1d HTF — Regime-Adaptive Strategy

Hypothesis: After analyzing 494+ failed strategies, the pattern is clear:
- Higher timeframes (12h, 1d) work better than lower TFs (1h, 30m)
- Static strategies fail in bear markets (2025 test period)
- Need REGIME-ADAPTIVE logic: mean-revert in chop, trend-follow otherwise
- #546 (12h chop + connors) had Return=+35.5% but Sharpe=-0.004 — close!
- This strategy combines: Choppiness Index regime + Connors RSI + HMA trend

Strategy Logic:
1. 1d HMA(21) for MAJOR trend bias (HTF direction filter)
2. 12h Choppiness Index(14) for regime detection:
   - CHOP > 61.8 = RANGE regime → use Connors RSI mean-reversion
   - CHOP < 38.2 = TREND regime → use HMA pullback entries
   - 38.2-61.8 = NEUTRAL → reduce position size or stay flat
3. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 20 in range regime + price > 1d HMA
   - Short: CRSI > 80 in range regime + price < 1d HMA
4. HMA Pullback (trend regime):
   - Long: price > HMA(21) + RSI(14) 35-55 pullback
   - Short: price < HMA(21) + RSI(14) 45-65 rally
5. ATR(14) 2.5x trailing stop on all positions
6. Position size: 0.30 base, 0.20 in neutral regime

Why this might beat Sharpe=0.435:
- Regime-adaptive = works in both bull (2021-2024) and bear (2025) markets
- 12h TF = 20-50 trades/year (per Rule 10), not too many for fee drag
- 1d HTF filter prevents major counter-trend losses
- Connors RSI has 75% win rate in range markets (proven in literature)
- Simpler than failed #548/#550 (no session/volume filters)

Position sizing: 0.30 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_chop_connors_hma_1d_v2"
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
    CHOP = 100 * (ATR(1, n) / (Highest High(n) - Lowest Low(n))) * 100 / sqrt(n)
    
    Interpretation:
    - CHOP > 61.8 = Range/Consolidation (mean-reversion regime)
    - CHOP < 38.2 = Trend (trend-following regime)
    - 38.2-61.8 = Neutral/Transition
    """
    n = period
    
    # ATR(1) = average true range with period 1 (just the TR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR(1) over n periods
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    
    # Highest High and Lowest Low over n periods
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    # CHOP formula
    chop = 100.0 * (atr_sum / price_range) / np.sqrt(n)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) on close prices - short-term momentum
    2. RSI(2) on streak lengths - duration of up/down moves
    3. PercentRank(100) - where close sits in recent range
    
    Interpretation:
    - CRSI < 10-20 = Oversold (long signal)
    - CRSI > 80-90 = Overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) on streak lengths
    # Streak = consecutive up or down days
    direction = np.sign(delta.values)
    direction[0] = 0
    
    streak = np.zeros(n)
    for i in range(1, n):
        if direction[i] == direction[i-1] and direction[i] != 0:
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 1 if direction[i] != 0 else 0
    
    # RSI on streak (treat streak as "gains" for up, "losses" for down)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100) - where close sits in recent range
    percent_rank = pd.Series(close).rolling(window=pr_period, min_periods=pr_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    # Combine into CRSI
    crsi = (rsi_close.values + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_48 = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_TREND = 0.30  # Full size in trending regime
    POSITION_SIZE_RANGE = 0.30  # Full size in range regime
    POSITION_SIZE_NEUTRAL = 0.15  # Half size in neutral regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_48[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS INDEX REGIME DETECTION ===
        chop_value = chop_14[i]
        
        if chop_value > 61.8:
            regime = 'range'  # Mean-reversion regime
            pos_size = POSITION_SIZE_RANGE
        elif chop_value < 38.2:
            regime = 'trend'  # Trend-following regime
            pos_size = POSITION_SIZE_TREND
        else:
            regime = 'neutral'  # Transition zone
            pos_size = POSITION_SIZE_NEUTRAL
        
        # === ENTRY LOGIC BY REGIME ===
        new_signal = 0.0
        
        if regime == 'range':
            # Mean-reversion using Connors RSI
            # Long: CRSI < 20 + price > 1d HMA (don't short in bull market)
            if crsi[i] < 20.0 and bull_regime_1d:
                new_signal = pos_size
            # Short: CRSI > 80 + price < 1d HMA (don't long in bear market)
            elif crsi[i] > 80.0 and bear_regime_1d:
                new_signal = -pos_size
        
        elif regime == 'trend':
            # Trend-following using HMA pullback + RSI filter
            # Long: price > HMA(21) + RSI pullback 35-55 + 1d bull
            rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
            if close[i] > hma_12h_21[i] and rsi_pullback_long and bull_regime_1d:
                if hma_1d_slope_bull:
                    new_signal = pos_size
                else:
                    new_signal = pos_size * 0.7
            
            # Short: price < HMA(21) + RSI rally 45-65 + 1d bear
            rsi_rally_short = 45.0 <= rsi_14[i] <= 65.0
            if close[i] < hma_12h_21[i] and rsi_rally_short and bear_regime_1d:
                if hma_1d_slope_bear:
                    new_signal = -pos_size
                else:
                    new_signal = -pos_size * 0.7
        
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
        
        # === EXIT CONDITIONS (regime flip or HTF flip) ===
        # Exit long on 1d regime flip to strong bear
        if in_position and position_side > 0:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to strong bull
        if in_position and position_side < 0:
            if bull_regime_1d and hma_1d_slope_bull:
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