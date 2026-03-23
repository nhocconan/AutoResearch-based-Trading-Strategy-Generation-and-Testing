#!/usr/bin/env python3
"""
Experiment #798: 30m Primary + 4h/1d HTF — Asymmetric Regime + Vol Spike Reversion + HTF Trend

Hypothesis: After 543 failed strategies, the key insight is:
1. 30m timeframe needs EXTREMELY strict filters to avoid fee drag (target 30-80 trades/year)
2. Asymmetric logic: different rules for bull vs bear regime (price vs SMA200)
3. Vol spike reversion (ATR7/ATR30 > 2.0) captures panic reversals with high win rate
4. 4h HMA(21) for trend direction, 1d Choppiness for regime (range vs trend)
5. Connors RSI components (RSI2 + RSI_Streak + PercentRank) for precise entry timing
6. 3+ confluence required: HTF trend + vol signal + RSI extreme + BB position
7. Position sizing: 0.20-0.25 (smaller for 30m to control drawdown)
8. Stoploss: 2.5x ATR trailing, exit on regime change

Why this might work where others failed:
- Most failed 30m strategies had too many trades (>200/year) → fee drag
- This uses 4h/1d for SIGNAL DIRECTION, 30m only for ENTRY TIMING
- Asymmetric logic prevents shorting in bull markets and longing in bear markets
- Vol spike reversion has proven 75%+ win rate in backtests
- Strict confluence (3+ filters) ensures only high-probability setups

Strategy design:
1. 4h HMA(21) aligned via mtf_data for primary trend bias
2. 1d Choppiness Index(14) for regime detection (CHOP>55=range, <45=trend)
3. 30m ATR ratio (ATR7/ATR30) for vol spike detection (>2.0 = extreme)
4. 30m Connors RSI components (RSI2, RSI_Streak2, PercentRank100)
5. 30m Bollinger Bands(20, 2.5) for mean reversion bounds
6. 30m Volume filter (>1.5x SMA20) for confirmation
7. Entry: need HTF trend + vol spike OR RSI extreme + BB breach + volume
8. Exit: opposite signal, stoploss, or regime change
9. Discrete signals: 0.0, ±0.20, ±0.25

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive
Timeframe: 30m (with 4h/1d HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_asym_vol_crsi_hma_4h1d_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches.
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
            tr_sum += max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi_components(close, rsi_period=2, streak_period=2, pr_period=100):
    """
    Connors RSI components for precise entry timing.
    CRSI = (RSI(close, 2) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10, Short: CRSI > 90
    """
    n = len(close)
    
    # RSI(2)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = 100 * up_streaks / streak_period
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / pr_period
        percent_rank[i] = 100 * rank
    
    return rsi_short, streak_rsi, percent_rank

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    atr_30m = calculate_atr(high, low, close, period=14)
    atr_30m_7 = calculate_atr(high, low, close, period=7)
    atr_30m_30 = calculate_atr(high, low, close, period=30)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.5)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    rsi_30m_14 = calculate_rsi(close, period=14)
    
    # Connors RSI components
    rsi_2, streak_rsi, percent_rank = calculate_connors_rsi_components(close)
    
    # Calculate SMA200 for bull/bear regime
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
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
        if np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_30m[i]) or vol_sma_30m[i] <= 1e-10:
            continue
        if np.isnan(rsi_30m_14[i]) or np.isnan(rsi_2[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(atr_30m_7[i]) or np.isnan(atr_30m_30[i]):
            continue
        
        # === BULL/BEAR REGIME (SMA200) ===
        bull_regime = close[i] > sma_200[i]
        bear_regime = close[i] < sma_200[i]
        
        # === TREND BIAS (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d_aligned[i] > 55
        trending_regime = chop_1d_aligned[i] < 45
        
        # === VOLUME CONFIRMATION (strict for 30m) ===
        volume_confirmed = volume[i] > 1.5 * vol_sma_30m[i]
        
        # === VOLATILITY SPIKE (ATR ratio) ===
        atr_ratio = atr_30m_7[i] / (atr_30m_30[i] + 1e-10)
        vol_spike = atr_ratio > 2.0
        vol_normal = atr_ratio < 1.2
        
        # === RSI SIGNALS (Connors RSI components) ===
        crsi = (rsi_2[i] + streak_rsi[i] + percent_rank[i]) / 3 if not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]) else 50
        rsi_extreme_oversold = rsi_30m_14[i] < 25
        rsi_extreme_overbought = rsi_30m_14[i] > 75
        crsi_oversold = crsi < 15
        crsi_overbought = crsi > 85
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        
        # Count confluence signals
        long_confluence = 0
        short_confluence = 0
        
        # === LONG SETUP (asymmetric: only in bull regime or strong reversal) ===
        if bull_regime:
            # Bull regime: look for pullback entries
            if trend_4h_bullish:
                long_confluence += 1
            if rsi_extreme_oversold or crsi_oversold:
                long_confluence += 1
            if below_bb_lower:
                long_confluence += 1
            if volume_confirmed:
                long_confluence += 1
            if vol_spike:
                long_confluence += 1  # Vol spike + oversold = reversal opportunity
        elif bear_regime:
            # Bear regime: only strong reversal signals
            if vol_spike and rsi_extreme_oversold and below_bb_lower:
                long_confluence += 2  # Strong reversal setup
            if crsi_oversold and below_bb_lower and volume_confirmed:
                long_confluence += 2
        
        # === SHORT SETUP (asymmetric: only in bear regime or strong reversal) ===
        if bear_regime:
            # Bear regime: look for rally entries
            if trend_4h_bearish:
                short_confluence += 1
            if rsi_extreme_overbought or crsi_overbought:
                short_confluence += 1
            if above_bb_upper:
                short_confluence += 1
            if volume_confirmed:
                short_confluence += 1
            if vol_spike:
                short_confluence += 1  # Vol spike + overbought = reversal opportunity
        elif bull_regime:
            # Bull regime: only strong reversal signals
            if vol_spike and rsi_extreme_overbought and above_bb_upper:
                short_confluence += 2  # Strong reversal setup
            if crsi_overbought and above_bb_upper and volume_confirmed:
                short_confluence += 2
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC (need 3+ confluence for 30m) ===
        if long_confluence >= 3:
            desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
        elif short_confluence >= 3:
            desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
        elif long_confluence >= 4:  # Very strong long without volume
            desired_signal = REDUCED_SIZE
        elif short_confluence >= 4:  # Very strong short without volume
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
                # Hold long if bull regime and trend intact
                if bull_regime and (trend_4h_bullish or rsi_30m_14[i] < 70):
                    desired_signal = BASE_SIZE if trend_4h_bullish else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if bear regime and trend intact
                if bear_regime and (trend_4h_bearish or rsi_30m_14[i] > 30):
                    desired_signal = -BASE_SIZE if trend_4h_bearish else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if regime changes to bear + RSI overbought
            if bear_regime and rsi_30m_14[i] > 65:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if regime changes to bull + RSI oversold
            if bull_regime and rsi_30m_14[i] < 35:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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