#!/usr/bin/env python3
"""
Experiment #856: 12h Primary + 1d HTF — CRSI Mean Reversion + Choppiness Regime

Hypothesis: After analyzing 592+ failed strategies, the key insight is:
1. 12h timeframe showed promise (#852 Sharpe=0.504) with CRSI + Choppiness
2. Lower TF strategies fail due to fee drag from too many trades
3. 12h targets 20-50 trades/year = optimal fee/trade balance
4. CRSI (Connors RSI) has proven 75% win rate on mean reversion
5. Choppiness Index cleanly separates range vs trend regimes
6. 1d HMA provides trend bias without over-filtering

Strategy design:
1. 12h Primary timeframe (target 30-50 trades/year)
2. 1d HMA(21) for trend bias (aligned properly via mtf_data)
3. Connors RSI(3,2,100) for entry timing
4. Choppiness Index(14) for regime: >50 = range, <50 = trend
5. RSI(14) extreme filter for confluence
6. ATR(14) trailing stop (2.5x)
7. Dual regime: mean revert in chop, breakout in trend
8. RELAXED entry thresholds to GUARANTEE trades on all symbols

Why this should work:
- 12h has worked before (#852 Sharpe=0.504)
- CRSI is proven mean-reversion indicator (75% win rate)
- Choppiness regime filter adapts to market conditions
- 1d HTF trend bias prevents counter-trend trades
- Relaxed thresholds ensure we get trades (not 0 like #845, #850, #855)

Target: Sharpe > 0.612, trades >= 20 train, >= 5 test, ALL symbols positive
Timeframe: 12h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_hma_1d_atr_v3"
timeframe = "12h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    
    Range 0-100. <10 = oversold, >90 = overbought.
    For 12h timeframe, use relaxed thresholds: <20 / >80
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    if len(streak_gain) >= streak_period:
        streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        streak_avg_gain = np.concatenate([[np.nan], streak_avg_gain])
        streak_avg_loss = np.concatenate([[np.nan], streak_avg_loss])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
            streak_rsi = 100 - (100 / (1 + streak_rs))
        streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank (100)
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        price_changes = np.diff(close[i-rank_period+1:i+1])
        current_change = close[i] - close[i-1]
        rank = np.sum(price_changes < current_change)
        pct_rank[i] = 100 * rank / len(price_changes) if len(price_changes) > 0 else 50
    
    # Combine CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 50 = ranging, CHOP < 50 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 50
        trending_regime = chop_12h[i] < 50
        
        # === CRSI SIGNALS (Relaxed for 12h timeframe) ===
        crsi_oversold = crsi_12h[i] < 20
        crsi_overbought = crsi_12h[i] > 80
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        crsi_neutral_low = 20 <= crsi_12h[i] < 40
        crsi_neutral_high = 60 < crsi_12h[i] <= 80
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_extreme_oversold = rsi_12h[i] < 25
        rsi_extreme_overbought = rsi_12h[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 50) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI oversold + RSI oversold + trend alignment (any)
            if crsi_oversold and rsi_oversold:
                if trend_1d_bullish or above_sma200:
                    desired_signal = BASE_SIZE
                else:
                    # Allow counter-trend in strong mean reversion setup
                    if crsi_extreme_oversold:
                        desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + RSI overbought + trend alignment (any)
            if crsi_overbought and rsi_overbought:
                if trend_1d_bearish or below_sma200:
                    desired_signal = -BASE_SIZE
                else:
                    # Allow counter-trend in strong mean reversion setup
                    if crsi_extreme_overbought:
                        desired_signal = -REDUCED_SIZE
            
            # Fallback: extreme CRSI alone (guarantees trades)
            if desired_signal == 0:
                if crsi_extreme_oversold:
                    desired_signal = REDUCED_SIZE
                if crsi_extreme_overbought:
                    desired_signal = -REDUCED_SIZE
            
            # Additional fallback: extreme RSI alone
            if desired_signal == 0:
                if rsi_extreme_oversold:
                    desired_signal = REDUCED_SIZE
                if rsi_extreme_overbought:
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 50) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback OR Donchian breakout
            if trend_1d_bullish or above_sma200:
                if crsi_neutral_low and rsi_oversold:
                    desired_signal = BASE_SIZE
                elif donchian_breakout_long:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + CRSI pullback OR Donchian breakout
            if trend_1d_bearish or below_sma200:
                if crsi_neutral_high and rsi_overbought:
                    desired_signal = -BASE_SIZE
                elif donchian_breakout_short:
                    desired_signal = -REDUCED_SIZE
            
            # Fallback: Donchian breakout alone in strong trend
            if desired_signal == 0:
                if donchian_breakout_long and (trend_1d_bullish or above_sma200):
                    desired_signal = REDUCED_SIZE
                if donchian_breakout_short and (trend_1d_bearish or below_sma200):
                    desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if (trend_1d_bullish or above_sma200) and crsi_12h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_1d_bearish or below_sma200) and crsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + CRSI overbought
            if trend_1d_bearish and below_sma200 and crsi_12h[i] > 80:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + CRSI oversold
            if trend_1d_bullish and above_sma200 and crsi_12h[i] < 20:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_12h[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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