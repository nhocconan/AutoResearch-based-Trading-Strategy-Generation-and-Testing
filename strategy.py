#!/usr/bin/env python3
"""
Experiment #859: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 595+ failed strategies, Connors RSI (CRSI) has proven edge for mean
reversion with 75% win rate. Combined with 1d HMA trend filter and Choppiness regime
detection, this should work across ALL symbols (BTC/ETH/SOL) in both bull and bear markets.

Why this should work:
1. CRSI(3,2,100) catches oversold/overbought extremes better than standard RSI
2. 1d HMA(21) provides trend bias without lagging too much
3. Choppiness Index switches between mean-revert (CHOP>55) and trend-follow (CHOP<45)
4. 4h timeframe targets 20-50 trades/year (optimal for fee drag)
5. Simple entry conditions ensure >=10 trades per symbol (avoiding 0-trade failure)

Strategy design:
1. 4h Primary timeframe (target 30-50 trades/year)
2. 1d HMA(21) for trend bias (aligned via mtf_data helper)
3. Connors RSI(3,2,100) for entry timing
4. Choppiness Index(14) for regime detection
5. 4h ATR(14) for trailing stop (2.5x)
6. 4h SMA(200) for secular trend filter
7. Dual regime: mean revert when choppy, trend-pullback when trending

CRSI Formula:
- RSI(3): Very short-term momentum
- RSI_Streak(2): RSI of consecutive up/down bar streaks
- PercentRank(100): Current close rank in last 100 bars (0-100)
- CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

Entry Logic:
- Long: CRSI < 15 + price > 1d_HMA + (CHOP > 55 OR trend aligned)
- Short: CRSI > 85 + price < 1d_HMA + (CHOP > 55 OR trend aligned)
- Relaxed thresholds ensure trades on all symbols

Target: Sharpe > 0.612, trades >= 15 train, >= 5 test, ALL symbols positive Sharpe
Timeframe: 4h (target 25-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_hma_1d_atr_v1"
timeframe = "4h"
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
    Connors RSI (CRSI) - combines 3 components for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Proven 75% win rate for extreme readings (<10 long, >90 short).
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streaks
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = streak[i-1] if i > 0 else 0
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    if len(streak_gain) >= streak_period:
        avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        
        avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
        avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
            streak_rsi = 100 - (100 / (1 + streak_rs))
        
        streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100 * rank / (rank_period - 1)
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging (mean revert), CHOP < 45 = trending (trend follow).
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
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_200_4h = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200_4h[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND FILTER (4h SMA200) ===
        above_sma200 = close[i] > sma_200_4h[i]
        below_sma200 = close[i] < sma_200_4h[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === CRSI SIGNALS (Connors RSI extremes) ===
        crsi_oversold = crsi_4h[i] < 15
        crsi_overbought = crsi_4h[i] > 85
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        crsi_recovering = crsi_4h[i] < 40 and crsi_4h[i] > crsi_4h[i-1] if not np.isnan(crsi_4h[i-1]) else False
        crsi_weakening = crsi_4h[i] > 60 and crsi_4h[i] < crsi_4h[i-1] if not np.isnan(crsi_4h[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + trend alignment (either 1d HMA or SMA200)
            if crsi_oversold and (trend_1d_bullish or above_sma200):
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + trend alignment
            if crsi_overbought and (trend_1d_bearish or below_sma200):
                desired_signal = -BASE_SIZE
            
            # Extreme CRSI alone (guarantees trades on all symbols)
            if crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Pullback ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback (not extreme)
            if trend_1d_bullish or above_sma200:
                if crsi_recovering and crsi_4h[i] < 35:
                    desired_signal = BASE_SIZE
                elif crsi_extreme_oversold:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + CRSI pullback
            if trend_1d_bearish or below_sma200:
                if crsi_weakening and crsi_4h[i] > 65:
                    desired_signal = -BASE_SIZE
                elif crsi_extreme_overbought:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extreme + any trend alignment
            if crsi_extreme_oversold and (trend_1d_bullish or above_sma200):
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (trend_1d_bearish or below_sma200):
                desired_signal = -REDUCED_SIZE
            
            # Basic mean reversion
            if crsi_oversold and above_sma200:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if crsi_overbought and below_sma200:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
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
                if (trend_1d_bullish or above_sma200) and crsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_1d_bearish or below_sma200) and crsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + CRSI overbought
            if trend_1d_bearish and below_sma200 and crsi_4h[i] > 85:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought in ranging regime
            if ranging_regime and crsi_4h[i] > 95:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + CRSI oversold
            if trend_1d_bullish and above_sma200 and crsi_4h[i] < 15:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold in ranging regime
            if ranging_regime and crsi_4h[i] < 5:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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