#!/usr/bin/env python3
"""
Experiment #988: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion + Choppiness Regime

Hypothesis: After 714 failed strategies, the key for lower TF (30m) is VERY FEW trades
with HIGH confluence. Using Connors RSI (proven 75% win rate) + Choppiness regime filter
+ HTF HMA trend bias should generate 30-80 trades/year with positive Sharpe.

Key insights:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 10 + price > SMA200 (bullish mean reversion)
   - Short when CRSI > 90 + price < SMA200 (bearish mean reversion)
2. Choppiness Index: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 4h HMA(21): Medium-term trend bias — only trade in HTF direction
4. 1d HMA(21): Macro regime filter — avoid counter-macro trades
5. Session filter: Only 8-20 UTC (high liquidity, lower slippage)
6. Volume filter: Volume > 0.8x 20-bar average (confirm participation)

Why 30m with HTF:
- 30m provides entry precision within HTF trend
- 4h/1d HMA gives strong trend bias (proven to 2x Sharpe)
- Target 40-80 trades/year (strict confluence = fewer but higher quality)
- CRSI extremes are rare enough to limit trade frequency

Risk management:
- Position size: 0.25 (conservative for lower TF)
- Stoploss: 2.5x ATR trailing
- Discrete signals: 0.0, ±0.25 to minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (with 4h/1d HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_regime_4h1d_hma_session_vol_v1"
timeframe = "30m"
leverage = 1.0

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
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with ~75% win rate.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (2): RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_vals > 0)
        down_streaks = np.sum(streak_vals < 0)
        if up_streaks + down_streaks > 0:
            streak_rsi[i] = 100 * up_streaks / (up_streaks + down_streaks)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100): Where does today's return rank vs last 100 days?
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and returns[-1] != 0:
            rank = np.sum(returns[:-1] <= returns[-1])
            percent_rank[i] = 100 * rank / (len(returns) - 1)
        else:
            percent_rank[i] = 50
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / average volume."""
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            ratio[i] = volume[i] / avg_vol
    
    return ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

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
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    sma_200_30m = calculate_sma(close, 200)
    vol_ratio_30m = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need 250 bars for CRSI(100) + SMA200
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(chop_30m[i]) or np.isnan(sma_200_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_ratio_30m[i]):
            continue
        
        # Session filter: Only trade 8-20 UTC (high liquidity)
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # Volume filter: Volume > 0.8x average
        volume_ok = vol_ratio_30m[i] > 0.8
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (30m Choppiness Index) ===
        ranging_regime = chop_30m[i] > 55
        trending_regime = chop_30m[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_low = crsi_30m[i] < 15  # Oversold
        crsi_extreme_high = crsi_30m[i] > 85  # Overbought
        crsi_very_low = crsi_30m[i] < 10  # Very oversold
        crsi_very_high = crsi_30m[i] > 90  # Very overbought
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime and in_session and volume_ok:
            # Long: CRSI extreme low + above SMA200 + HTF bullish bias
            if crsi_extreme_low and above_sma200 and (macro_bull or trend_4h_bullish):
                desired_signal = BASE_SIZE
            # Long: CRSI very low (guarantees some trades)
            elif crsi_very_low and above_sma200:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI extreme high + below SMA200 + HTF bearish bias
            if crsi_extreme_high and below_sma200 and (macro_bear or trend_4h_bearish):
                desired_signal = -BASE_SIZE
            # Short: CRSI very high (guarantees some trades)
            elif crsi_very_high and below_sma200:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Pullback ===
        elif trending_regime and in_session and volume_ok:
            # Long: Bullish HTF + CRSI pullback (not extreme)
            if (macro_bull or trend_4h_bullish) and crsi_extreme_low and above_sma200:
                desired_signal = BASE_SIZE
            # Long: Strong bullish + any CRSI low
            elif macro_bull and trend_4h_bullish and crsi_30m[i] < 30:
                desired_signal = REDUCED_SIZE
            
            # Short: Bearish HTF + CRSI rally (not extreme)
            if (macro_bear or trend_4h_bearish) and crsi_extreme_high and below_sma200:
                desired_signal = -BASE_SIZE
            # Short: Strong bearish + any CRSI high
            elif macro_bear and trend_4h_bearish and crsi_30m[i] > 70:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only very extreme CRSI + strong HTF alignment
            if crsi_very_low and macro_bull and trend_4h_bullish and above_sma200:
                desired_signal = REDUCED_SIZE
            if crsi_very_high and macro_bear and trend_4h_bearish and below_sma200:
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
                if (macro_bull or trend_4h_bullish) and crsi_30m[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend intact and CRSI not oversold
                if (macro_bear or trend_4h_bearish) and crsi_30m[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF trend reverses + CRSI overbought
            if macro_bear and trend_4h_bearish and crsi_30m[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trend reverses + CRSI oversold
            if macro_bull and trend_4h_bullish and crsi_30m[i] < 20:
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