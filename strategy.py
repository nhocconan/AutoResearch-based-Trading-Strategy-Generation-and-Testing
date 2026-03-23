#!/usr/bin/env python3
"""
Experiment #852: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 562+ failed strategies, the key insight for 12h timeframe is:
1. Connors RSI (CRSI) has documented 75% win rate for mean reversion entries
2. Choppiness Index cleanly separates ranging vs trending regimes
3. 12h TF naturally limits trades to 20-50/year (fee-efficient)
4. HMA(21) on 1w provides long-term bias without overfitting

Strategy design:
1. 12h Primary timeframe (target 30-50 trades/year)
2. 1d HMA(21) for medium-term trend bias
3. 1w HMA(21) for long-term trend bias (meta-filter only)
4. Connors RSI(3,2,100) for entry timing — extremes <10/>90
5. Choppiness Index(14) for regime — CHOP>55 mean revert, CHOP<45 trend follow
6. 12h ATR(14) for trailing stop (2.5x)
7. Relaxed entry thresholds to ensure trades on ALL symbols (BTC/ETH/SOL)

Why this should work:
- CRSI captures oversold/overbought better than standard RSI
- CHOP regime filter avoids mean-reversion in strong trends
- 12h TF = fewer trades = less fee drag than 1h/4h strategies
- Dual HTF (1d + 1w) provides trend confluence without look-ahead

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols Sharpe > 0
Timeframe: 12h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_hma_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average — smoother than EMA, less lag than SMA."""
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Long: CRSI < 10 (extreme oversold)
    Short: CRSI > 90 (extreme overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak Length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and close[i-1] >= close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and close[i-1] <= close[i-2] else -1
        else:
            streak[i] = streak[i-1] if i > 0 else 0
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    
    # Component 3: Percent Rank of price change over rank_period
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        current_return = returns[-1] if len(returns) > 0 else 0
        rank = np.sum(returns[:-1] < current_return) if len(returns) > 1 else 0
        pct_rank[i] = 100 * rank / max(len(returns) - 1, 1)
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
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
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        crsi_oversold = crsi_12h[i] < 25
        crsi_overbought = crsi_12h[i] > 75
        crsi_rising = crsi_12h[i] > crsi_12h[i-1] if not np.isnan(crsi_12h[i-1]) else False
        crsi_falling = crsi_12h[i] < crsi_12h[i-1] if not np.isnan(crsi_12h[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI extreme oversold + ANY trend alignment (relaxed for trade generation)
            if crsi_extreme_oversold:
                if trend_1w_bullish or trend_1d_bullish or above_sma200:
                    desired_signal = BASE_SIZE
                else:
                    # Allow entry even without trend alignment in strong oversold
                    if crsi_12h[i] < 10:
                        desired_signal = REDUCED_SIZE
            
            # Short: CRSI extreme overbought + ANY trend alignment
            if crsi_extreme_overbought:
                if trend_1w_bearish or trend_1d_bearish or below_sma200:
                    desired_signal = -BASE_SIZE
                else:
                    # Allow entry even without trend alignment in strong overbought
                    if crsi_12h[i] > 90:
                        desired_signal = -REDUCED_SIZE
            
            # CRSI recovery from oversold (rising from extreme)
            if crsi_rising and crsi_oversold and crsi_12h[i-1] < 20:
                if desired_signal == 0:
                    desired_signal = REDUCED_SIZE
            
            # CRSI decline from overbought (falling from extreme)
            if crsi_falling and crsi_overbought and crsi_12h[i-1] > 80:
                if desired_signal == 0:
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback (not extreme)
            if trend_1d_bullish and trend_1w_bullish:
                if crsi_oversold and crsi_rising:
                    desired_signal = BASE_SIZE
                elif above_sma200 and crsi_12h[i] < 40:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + CRSI rally (not extreme)
            if trend_1d_bearish and trend_1w_bearish:
                if crsi_overbought and crsi_falling:
                    desired_signal = -BASE_SIZE
                elif below_sma200 and crsi_12h[i] > 60:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require trend alignment + CRSI extreme
            if crsi_extreme_oversold and (trend_1d_bullish or above_sma200):
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (trend_1d_bearish or below_sma200):
                desired_signal = -REDUCED_SIZE
            
            # Moderate CRSI with strong trend confluence
            if crsi_oversold and trend_1d_bullish and trend_1w_bullish:
                if desired_signal == 0:
                    desired_signal = REDUCED_SIZE
            
            if crsi_overbought and trend_1d_bearish and trend_1w_bearish:
                if desired_signal == 0:
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
                if (trend_1d_bullish or trend_1w_bullish) and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_1d_bearish or trend_1w_bearish) and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HTF trends reverse + CRSI overbought
            if trend_1d_bearish and trend_1w_bearish and crsi_12h[i] > 85:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought in ranging regime
            if ranging_regime and crsi_12h[i] > 90:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HTF trends reverse + CRSI oversold
            if trend_1d_bullish and trend_1w_bullish and crsi_12h[i] < 15:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold in ranging regime
            if ranging_regime and crsi_12h[i] < 10:
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