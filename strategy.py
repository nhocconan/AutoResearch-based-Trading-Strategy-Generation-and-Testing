#!/usr/bin/env python3
"""
Experiment #576: 12h Primary + 1d HTF — Dual Regime (Choppiness + Connors RSI + HMA)

Hypothesis: After 500+ failed experiments, the pattern is clear:
- Single-regime strategies fail (trend-only gets whipsawed in 2022, MR-only misses big moves)
- 12h timeframe with 1d HTF showed promise (ETH Sharpe +0.923 with Choppiness + Connors)
- Need DUAL REGIME: trend-follow when CHOP < 38.2, mean-revert when CHOP > 61.8
- Connors RSI (CRSI) has 75% win rate for MR entries at extremes (<10 long, >90 short)
- 1d HMA(21) for major bias - don't MR against daily trend
- Target: 20-50 trades/year on 12h (per Rule 10), Sharpe > 0.435 (current best)

Key innovations vs failed attempts:
1. CHOPPINESS INDEX regime filter (not used in most failed strats)
2. CONNORS RSI for MR (RSI3 + RSI_Streak2 + PercentRank100) / 3
3. Dual-mode logic: different entry rules per regime
4. 1d HTF HMA for bias (not 4h which is too close to 12h)
5. Conservative sizing: 0.25 base, 0.30 for strong confluence
6. ATR 2.5x trailing stop + regime-flip exit

Position sizing: 0.25 base, 0.30 max (discrete per Rule 4)
Stoploss: 2.5 * ATR(14) trailing
Target: >=30 trades/symbol train, >=3 test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_1d_dual_v1"
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
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = period
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR over period
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Avoid division by zero
    range_value = highest_high - lowest_low
    range_value = np.where(range_value < 1e-10, 1e-10, range_value)
    
    # CHOP formula
    chop = 100.0 * np.log10(atr_sum / range_value) / np.log10(n)
    
    # Clamp to valid range
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period_rsi)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    returns = close_s.pct_change()
    streak = np.zeros(n)
    
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values (treating streak as "price")
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # Component 3: PercentRank of returns
    percent_rank = np.zeros(n) * np.nan
    
    for i in range(period_rank, n):
        window_returns = returns.iloc[i-period_rank+1:i+1].dropna()
        if len(window_returns) > 0:
            current_return = returns.iloc[i]
            percent_rank[i] = (window_returns <= current_return).sum() / len(window_returns) * 100.0
    
    # Combine components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    
    # 12h HMA for local trend
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_STRONG = 0.30
    
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
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        
        # === 1D MAJOR TREND BIAS (HTF filter) ===
        bull_bias_1d = close[i] > hma_1d_21_aligned[i]
        bear_bias_1d = close[i] < hma_1d_21_aligned[i]
        strong_bull_1d = bull_bias_1d and (hma_1d_21_aligned[i] > hma_1d_50_aligned[i])
        strong_bear_1d = bear_bias_1d and (hma_1d_21_aligned[i] < hma_1d_50_aligned[i])
        
        # === 12H LOCAL TREND ===
        bull_12h = close[i] > hma_12h_21[i]
        bear_12h = close[i] < hma_12h_21[i]
        strong_bull_12h = bull_12h and (hma_12h_21[i] > hma_12h_50[i])
        strong_bear_12h = bear_12h and (hma_12h_21[i] < hma_12h_50[i])
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_14[i] > 61.8  # Range/mean-reversion mode
        trending_regime = chop_14[i] < 38.2  # Trend-follow mode
        neutral_regime = not choppy_regime and not trending_regime
        
        # === DUAL REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: MEAN REVERSION (choppy regime, CHOP > 61.8)
        if choppy_regime:
            # Long: CRSI < 15 (oversold) + 1d bias not strongly bear
            if crsi[i] < 15 and not strong_bear_1d:
                if bull_bias_1d:
                    new_signal = POSITION_SIZE_STRONG  # 0.30 with 1d bull bias
                else:
                    new_signal = POSITION_SIZE_BASE  # 0.25 neutral 1d
            
            # Short: CRSI > 85 (overbought) + 1d bias not strongly bull
            elif crsi[i] > 85 and not strong_bull_1d:
                if bear_bias_1d:
                    new_signal = -POSITION_SIZE_STRONG  # -0.30 with 1d bear bias
                else:
                    new_signal = -POSITION_SIZE_BASE  # -0.25 neutral 1d
        
        # MODE 2: TREND FOLLOW (trending regime, CHOP < 38.2)
        elif trending_regime:
            # Long: 12h bull + 1d bull bias + ADX confirms trend
            if strong_bull_12h and bull_bias_1d and adx_14[i] > 20:
                new_signal = POSITION_SIZE_STRONG
            
            # Short: 12h bear + 1d bear bias + ADX confirms trend
            elif strong_bear_12h and bear_bias_1d and adx_14[i] > 20:
                new_signal = -POSITION_SIZE_STRONG
            
            # Weaker trend entries (one TF confirms)
            elif bull_12h and bull_bias_1d:
                new_signal = POSITION_SIZE_BASE
            elif bear_12h and bear_bias_1d:
                new_signal = -POSITION_SIZE_BASE
        
        # MODE 3: NEUTRAL (38.2 <= CHOP <= 61.8) - reduced activity
        elif neutral_regime:
            # Only enter on strong confluence
            if strong_bull_12h and strong_bull_1d and crsi[i] < 40:
                new_signal = POSITION_SIZE_BASE
            elif strong_bear_12h and strong_bear_1d and crsi[i] > 60:
                new_signal = -POSITION_SIZE_BASE
        
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
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        # Exit long on 1d bias flip to strong bear
        if in_position and position_side > 0:
            if strong_bear_1d:
                new_signal = 0.0
        
        # Exit short on 1d bias flip to strong bull
        if in_position and position_side < 0:
            if strong_bull_1d:
                new_signal = 0.0
        
        # Exit MR positions when CRSI mean-reverts
        if in_position and position_side > 0 and choppy_regime:
            if crsi[i] > 50:  # CRSI recovered from oversold
                new_signal = 0.0
        
        if in_position and position_side < 0 and choppy_regime:
            if crsi[i] < 50:  # CRSI recovered from overbought
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