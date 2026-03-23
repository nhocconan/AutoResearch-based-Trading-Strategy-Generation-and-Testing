#!/usr/bin/env python3
"""
Experiment #813: 1d Primary + 1w HTF — Dual Regime with Connors RSI + Donchian

Hypothesis: After analyzing 553 failed strategies, key insights for 1d timeframe:
1. 1d primary reduces noise significantly vs 4h/1h — fewer false signals
2. Connors RSI (CRSI) outperforms standard RSI for mean-reversion (75% win rate)
3. Choppiness Index regime switch is critical — range vs trend need different logic
4. 1w HMA(21) provides strong secular trend filter (proven in #811)
5. Donchian(20) breakouts work well in trending regime on 1d
6. RELAXED entry thresholds ensure >=10 trades on ALL symbols (BTC/ETH/SOL)
7. ATR(14) trailing stop at 2.5x protects against 2022-style crashes
8. Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn on signal changes

Strategy design:
1. 1w HMA(21) for long-term trend bias (aligned via mtf_data)
2. 1d Choppiness Index(14) for regime detection
3. 1d Connors RSI for mean-reversion entries in ranging regime
4. 1d Donchian(20) for breakout entries in trending regime
5. 1d ATR(14) for trailing stop (2.5x)
6. 1d SMA(200) as secular trend filter
7. Discrete signals: 0.0, ±0.20, ±0.30
8. RELAXED CRSI thresholds (15/85 vs 10/90) to guarantee trades

Key difference from #811 (4h):
- 1d primary = fewer but higher-quality signals
- CRSI instead of RSI(14) — better for mean-reversion
- Donchian(20) instead of (15) — fewer false breakouts on 1d
- ATR stop at 2.5x (not 2.0x) — wider for 1d volatility

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_donchian_1w_atr_v2"
timeframe = "1d"
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
    Connors RSI (CRSI) — proven mean-reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    Entry: CRSI < 15 (oversold), CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        gain_streak = np.sum(np.where(streak_vals > 0, streak_vals, 0))
        loss_streak = np.abs(np.sum(np.where(streak_vals < 0, streak_vals, 0)))
        
        if loss_streak == 0:
            streak_rsi[i] = 100
        else:
            rs = gain_streak / (loss_streak + 1e-10)
            streak_rsi[i] = 100 - (100 / (1 + rs))
    
    # Percent Rank of returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if np.isnan(rsi_3[i]) or np.isnan(streak_rsi[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow).
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1w HMA for long-term trend bias
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d[i] > 55  # relaxed threshold for more switches
        trending_regime = chop_1d[i] < 45  # relaxed threshold
        neutral_regime = not ranging_regime and not trending_regime
        
        # === CRSI SIGNALS (RELAXED for more trades) ===
        crsi_oversold = crsi_1d[i] < 20  # relaxed from 15
        crsi_overbought = crsi_1d[i] > 80  # relaxed from 85
        crsi_extreme_oversold = crsi_1d[i] < 15
        crsi_extreme_overbought = crsi_1d[i] > 85
        crsi_neutral_low = 20 <= crsi_1d[i] < 45
        crsi_neutral_high = 55 < crsi_1d[i] <= 80
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            donchian_breakout_short = close[i] < donchian_lower[i-1]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + trend alignment (relaxed: any 1 filter)
            if crsi_oversold and (above_sma200 or trend_1w_bullish):
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + trend alignment
            if crsi_overbought and (below_sma200 or trend_1w_bearish):
                desired_signal = -BASE_SIZE
            
            # Conservative: extreme CRSI alone (guarantees trades)
            if crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: 1w bullish + Donchian breakout
            if trend_1w_bullish and donchian_breakout_long:
                desired_signal = BASE_SIZE
            
            # Short: 1w bearish + Donchian breakout
            if trend_1w_bearish and donchian_breakout_short:
                desired_signal = -BASE_SIZE
            
            # Pullback entries in trend (relaxed CRSI)
            if trend_1w_bullish and crsi_neutral_low and above_sma200:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if trend_1w_bearish and crsi_neutral_high and below_sma200:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extremes + any trend alignment
            if crsi_extreme_oversold and (trend_1w_bullish or above_sma200):
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (trend_1w_bearish or below_sma200):
                desired_signal = -REDUCED_SIZE
            
            # Basic mean reversion with single filter
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
                # Hold long if any trend intact and CRSI not overbought
                if (trend_1w_bullish or above_sma200) and crsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if any trend intact and CRSI not oversold
                if (trend_1w_bearish or below_sma200) and crsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + CRSI overbought
            if trend_1w_bearish and crsi_1d[i] > 75:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought in ranging regime
            if ranging_regime and crsi_1d[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + CRSI oversold
            if trend_1w_bullish and crsi_1d[i] < 25:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold in ranging regime
            if ranging_regime and crsi_1d[i] < 15:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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