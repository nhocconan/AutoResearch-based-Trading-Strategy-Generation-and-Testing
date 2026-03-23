#!/usr/bin/env python3
"""
Experiment #1146: 12h Primary + 1d HTF — Dual Regime CRSI + Choppiness + HMA

Hypothesis: After 800+ failed experiments, the winning pattern is REGIME-ADAPTIVE logic.
BTC/ETH behave differently in trending vs ranging markets. This strategy:
1. Uses 1d HMA(21) for macro trend direction (proven across all symbols)
2. Uses 12h Choppiness Index(14) to detect regime: CHOP>55=range, CHOP<45=trend
3. In TREND regime: Follow 1d HMA direction with 12h RSI(14) pullback entries
4. In RANGE regime: Mean revert with Connors RSI extremes (CRSI<15 long, >85 short)
5. ATR(14) 2.5x trailing stoploss on all positions
6. Position size 0.28 discrete (minimize fee churn while maintaining exposure)

Why this should beat Sharpe=0.612:
- Dual regime adapts to market conditions (2022 crash = trend, 2025 = range)
- CRSI captures sharp reversals better than standard RSI
- Choppiness filter prevents trend-following in chop (major source of losses)
- 12h timeframe = 20-50 trades/year target (optimal for fee drag)
- Simpler than triple-regime approaches that generated 0 trades

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.28 base (discrete: 0.0, ±0.28)
Stoploss: 2.5x ATR trailing
Target: 25-50 trades/year, Sharpe > 0.612, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_chop_hma_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite mean reversion indicator.
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10-15, Short when CRSI > 85-90
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    diff = np.diff(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if diff[i-1] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif diff[i-1] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    streak_rsi_raw = np.zeros(n)
    for i in range(streak_period, n):
        if abs_streak[i] >= streak_period:
            streak_rsi_raw[i] = 100.0 if streak[i] > 0 else 0.0
        else:
            streak_rsi_raw[i] = 50.0 + streak[i] * 25.0
            streak_rsi_raw[i] = np.clip(streak_rsi_raw[i], 0, 100)
    
    streak_rsi = pd.Series(streak_rsi_raw).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1) * 100.0
        percent_rank[i] = rank
    
    # Combine into CRSI
    mask = (~np.isnan(rsi_short)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = ranging market (mean revert)
        # CHOP < 45 = trending market (trend follow)
        # 45-55 = transition zone (use trend logic)
        is_ranging = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        desired_signal = 0.0
        
        # === TREND REGIME LOGIC ===
        if is_trending or (not is_ranging and not is_trending):
            # Follow macro trend with RSI pullback entries
            # Long: macro bull + RSI pullback (40-50 range)
            if macro_bull and 38.0 < rsi_12h[i] < 52.0:
                desired_signal = BASE_SIZE
            
            # Short: macro bear + RSI pullback (48-62 range)
            elif macro_bear and 48.0 < rsi_12h[i] < 62.0:
                desired_signal = -BASE_SIZE
        
        # === RANGE REGIME LOGIC ===
        if is_ranging:
            # Mean reversion with CRSI extremes
            # Long: CRSI < 15 (oversold)
            if crsi[i] < 15.0:
                desired_signal = BASE_SIZE
            
            # Short: CRSI > 85 (overbought)
            elif crsi[i] > 85.0:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull OR ranging with no CRSI extreme
                if macro_bull or (not is_ranging):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bear OR ranging with no CRSI extreme
                if macro_bear or (not is_ranging):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses strongly
        if in_position and position_side > 0:
            # Exit long if macro reverses to bear AND trending regime
            if macro_bear and is_trending:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses to bull AND trending regime
            if macro_bull and is_trending:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals