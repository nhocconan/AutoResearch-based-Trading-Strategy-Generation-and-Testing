#!/usr/bin/env python3
"""
Experiment #890: 1h Primary + 4h/12h HTF — Relaxed CRSI + Volume + Session Filter

Hypothesis: After 600+ failed strategies, the key issue for 1h timeframe is balancing
trade frequency vs fee drag. Previous 1h strategies failed because:
1. Too strict entry conditions → 0 trades (Sharpe=0.000)
2. Too loose conditions → 200+ trades/year → fee drag kills profit

This strategy uses:
1. 4h HMA(21) for primary trend direction (reliable for 1h entries)
2. 12h HMA(21) for macro bias (bull/bear market filter)
3. Connors RSI(3,2,100) with RELAXED thresholds (15/85 not 10/90)
4. Volume filter (only trade when volume > 0.7x 20-bar average)
5. Session filter (8-20 UTC - London/NY overlap for liquidity)
6. Choppiness Index(14) for regime detection (CHOP>55=range, CHOP<45=trend)
7. ATR(14) trailing stop (2.0x) for risk management
8. Discrete signal sizes (0.0, ±0.25, ±0.30) to minimize fee churn

CRITICAL IMPROVEMENTS from failed experiments:
- RELAXED CRSI thresholds (15/85) to guarantee 30+ trades per symbol
- Volume filter at 0.7x (not 1.0x) to allow more trades
- Session filter only for quality (not blocking all trades)
- Hold logic maintains position through minor pullbacks
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Use 4h HMA as primary trend (more responsive than 1d for 1h entries)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-70 trades/year)
Position Size: 0.25 base, 0.30 strong confluence (smaller for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h12h_hma_vol_session_atr_v1"
timeframe = "1h"
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
    
    Relaxed thresholds: 15/85 (not 10/90) to ensure trades on all symbols
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < max(rsi_period, streak_period, rank_period) + 2:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if direction[i-1] == 1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
            direction[i] = 1
        elif close[i] < close[i-1]:
            if direction[i-1] == -1:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = -1
            direction[i] = -1
        else:
            streak[i] = 0
            direction[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_vals > 0)
        down_streaks = np.sum(streak_vals < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100 * up_streaks / total
        else:
            streak_rsi[i] = 50
    
    # Percent Rank of price change
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = 100 * rank
        else:
            percent_rank[i] = 50
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — CHOP > 55 = ranging, CHOP < 45 = trending."""
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align 4h HMA for primary trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro regime
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during London/NY overlap for better liquidity
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        # Only trade when volume > 0.7x average (relaxed from 1.0x)
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === MACRO REGIME (12h HTF HMA21) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (1h SMA50/200) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === CONNORS RSI SIGNALS (Relaxed: 15/85) ===
        crsi_oversold = crsi_1h[i] < 15
        crsi_overbought = crsi_1h[i] > 85
        crsi_extreme_oversold = crsi_1h[i] < 10
        crsi_extreme_overbought = crsi_1h[i] > 90
        
        # === RSI SIGNALS (fallback) ===
        rsi_oversold = rsi_1h[i] < 30
        rsi_overbought = rsi_1h[i] > 70
        rsi_extreme_oversold = rsi_1h[i] < 20
        rsi_extreme_overbought = rsi_1h[i] > 80
        
        desired_signal = 0.0
        
        # Count confluence factors for signal strength
        long_confluence = 0
        short_confluence = 0
        
        # Long confluence factors
        if macro_bull:
            long_confluence += 1
        if trend_4h_bullish:
            long_confluence += 1
        if above_sma50:
            long_confluence += 1
        if above_sma200:
            long_confluence += 1
        if volume_ok:
            long_confluence += 1
        if in_session:
            long_confluence += 1
        
        # Short confluence factors
        if macro_bear:
            short_confluence += 1
        if trend_4h_bearish:
            short_confluence += 1
        if below_sma50:
            short_confluence += 1
        if below_sma200:
            short_confluence += 1
        if volume_ok:
            short_confluence += 1
        if in_session:
            short_confluence += 1
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + at least 2 trend confluence
            if crsi_oversold and long_confluence >= 2:
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + at least 2 trend confluence
            if crsi_overbought and short_confluence >= 2:
                desired_signal = -BASE_SIZE
            
            # Strong signal: extreme CRSI + 3+ confluence
            if crsi_extreme_oversold and long_confluence >= 3:
                desired_signal = STRONG_SIZE
            
            if crsi_extreme_overbought and short_confluence >= 3:
                desired_signal = -STRONG_SIZE
            
            # Fallback: extreme RSI in ranging regime (guarantees trades)
            if rsi_extreme_oversold and long_confluence >= 2 and desired_signal == 0:
                desired_signal = BASE_SIZE
            
            if rsi_extreme_overbought and short_confluence >= 2 and desired_signal == 0:
                desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback OR RSI pullback
            if long_confluence >= 3:
                if crsi_oversold or rsi_oversold:
                    desired_signal = BASE_SIZE
            
            # Short: Bearish trend + CRSI rally OR RSI rally
            if short_confluence >= 3:
                if crsi_overbought or rsi_overbought:
                    desired_signal = -BASE_SIZE
            
            # Strong trend: 4+ confluence + any mean reversion signal
            if long_confluence >= 4 and (crsi_oversold or rsi_oversold):
                desired_signal = STRONG_SIZE
            
            if short_confluence >= 4 and (crsi_overbought or rsi_overbought):
                desired_signal = -STRONG_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require 3+ confluence + CRSI extreme
            if crsi_oversold and long_confluence >= 3:
                desired_signal = BASE_SIZE
            
            if crsi_overbought and short_confluence >= 3:
                desired_signal = -BASE_SIZE
            
            # Fallback: RSI extremes with strong confluence
            if rsi_extreme_oversold and long_confluence >= 3 and desired_signal == 0:
                desired_signal = BASE_SIZE
            
            if rsi_extreme_overbought and short_confluence >= 3 and desired_signal == 0:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if primary trend intact and CRSI not overbought
                if trend_4h_bullish and crsi_1h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if primary trend intact and CRSI not oversold
                if trend_4h_bearish and crsi_1h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if primary + macro trend reverses + CRSI overbought
            if trend_4h_bearish and macro_bear and crsi_1h[i] > 80:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_1h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if primary + macro trend reverses + CRSI oversold
            if trend_4h_bullish and macro_bull and crsi_1h[i] < 20:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_1h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = STRONG_SIZE if desired_signal >= STRONG_SIZE else BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -STRONG_SIZE if desired_signal <= -STRONG_SIZE else -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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