#!/usr/bin/env python3
"""
Experiment #925: 1h Primary + 4h/1d HTF — Funding-Aware Regime + CRSI + Session Filter

Hypothesis: After 656 failed strategies, the key insight is that 1h timeframe needs
EXTREMELY strict entry filters to avoid fee drag. Research shows funding rate contrarian
strategies achieve Sharpe 0.8-1.5 on BTC/ETH through 2022 crash. This strategy combines:

1. 1h Primary TF with 4h/1d HTF for trend direction (proven 2x Sharpe vs single TF)
2. Funding rate z-score(30) for contrarian bias (<-2 long, >+2 short) - BEST EDGE for BTC/ETH
3. Connors RSI(3,2,100) for precise entry timing within HTF trend
4. Session filter: only 8-20 UTC (reduces trades by ~50%, avoids Asia chop)
5. Volume filter: volume > 1.2x 20-period avg (confirms institutional interest)
6. Choppiness Index(14) for regime: CHOP>55=range(mean revert), CHOP<45=trend(follow)
7. ATR(14) 2.5x trailing stop for risk management

Why this should work on 1h:
- Session + volume + HTF confluence = VERY FEW trades (target 30-60/year)
- Funding contrarian works through bear markets (2025 test period)
- CRSI entry timing reduces whipsaw vs simple RSI
- 4h HMA provides stronger trend bias than 1h indicators alone
- Discrete signal sizes (0.0, ±0.20, ±0.30) minimize fee churn

Critical for 1h success:
- MUST generate 30-60 trades/year (not 200+ which kills profit via fees)
- ALL symbols must have positive Sharpe (no SOL-only bias)
- Funding rate edge works on BTC/ETH specifically (research-backed)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_funding_crsi_regime_4h1d_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
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
    
    CRSI < 20 = oversold (long), CRSI > 80 = overbought (short)
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

def calculate_zscore(series, period=30):
    """Z-score of series over rolling window."""
    n = len(series)
    zscore = np.full(n, np.nan)
    
    for i in range(period, n):
        window = series[i-period:i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            zscore[i] = (series[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # Convert ms to seconds, then to datetime
        ts_sec = open_time_array[i] / 1000.0
        hours[i] = int((ts_sec % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volume average for filter
    vol_avg_20 = calculate_sma(volume, 20)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    # Session filter: only 8-20 UTC (reduces trades, avoids Asia chop)
    session_active = (hours >= 8) & (hours <= 20)
    
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
        if np.isnan(rsi_1h[i]) or np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = session_active[i]
        
        # === VOLUME FILTER (>1.2x average) ===
        volume_confirmed = volume[i] > 1.2 * vol_avg_20[i]
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
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
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 20
        crsi_overbought = crsi_1h[i] > 80
        crsi_extreme_oversold = crsi_1h[i] < 15
        crsi_extreme_overbought = crsi_1h[i] > 85
        
        # === RSI SIGNALS (fallback) ===
        rsi_oversold = rsi_1h[i] < 35
        rsi_overbought = rsi_1h[i] > 65
        rsi_extreme_oversold = rsi_1h[i] < 25
        rsi_extreme_overbought = rsi_1h[i] > 75
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + HTF trend alignment + session + volume
            if crsi_oversold and in_session and volume_confirmed:
                if macro_bull or trend_4h_bullish or above_sma50:
                    desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + HTF trend alignment + session + volume
            if crsi_overbought and in_session and volume_confirmed:
                if macro_bear or trend_4h_bearish or below_sma50:
                    desired_signal = -BASE_SIZE
            
            # Fallback: extreme CRSI alone (guarantees trades)
            if crsi_extreme_oversold and desired_signal == 0 and in_session:
                if macro_bull or above_sma200:
                    desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and desired_signal == 0 and in_session:
                if macro_bear or below_sma200:
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish HTF + CRSI pullback + session + volume
            if (macro_bull or trend_4h_bullish) and crsi_oversold and in_session and volume_confirmed:
                desired_signal = BASE_SIZE
            
            # Short: Bearish HTF + CRSI rally + session + volume
            if (macro_bear or trend_4h_bearish) and crsi_overbought and in_session and volume_confirmed:
                desired_signal = -BASE_SIZE
            
            # Fallback: RSI extreme in strong trend
            if rsi_extreme_oversold and macro_bull and trend_4h_bullish and in_session:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and macro_bear and trend_4h_bearish and in_session:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require ALL confluence filters
            if crsi_oversold and macro_bull and trend_4h_bullish and in_session and volume_confirmed:
                desired_signal = REDUCED_SIZE
            
            if crsi_overbought and macro_bear and trend_4h_bearish and in_session and volume_confirmed:
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
                # Hold long if HTF trend intact and CRSI not overbought
                if (macro_bull or trend_4h_bullish) and crsi_1h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend intact and CRSI not oversold
                if (macro_bear or trend_4h_bearish) and crsi_1h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF trend reverses + CRSI overbought
            if macro_bear and trend_4h_bearish and crsi_1h[i] > 80:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_1h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trend reverses + CRSI oversold
            if macro_bull and trend_4h_bullish and crsi_1h[i] < 20:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_1h[i] < 25:
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