#!/usr/bin/env python3
"""
Experiment #225: 1h Primary + 4h/1d HTF — CRSI Mean Reversion + Choppiness Regime + HMA Trend

Hypothesis: After multiple failures with complex regime switching (#213, #214, #222, #223) and
0-trade disasters (#215, #218), simplify to proven Connors RSI mean reversion within HTF trend.
CRSI has 75% win rate in research. Use Choppiness Index to avoid trading in unclear regimes.
4h HMA provides macro trend bias. 1h CRSI provides entry timing.

Key design choices to avoid past failures:
1. CRSI thresholds LOOSE enough to generate trades (long: CRSI<20, short: CRSI>80)
2. Choppiness filter NOT too strict (CHOP>50 = range mode, CHOP<40 = trend mode)
3. 4h HMA alignment via mtf_data helper (NO manual resampling)
4. Position size: 0.25 full, 0.15 half (discrete to minimize fee churn)
5. ATR stoploss: 2.5x trailing stop
6. Target: 40-70 trades/year on 1h (use session filter 8-20 UTC for liquidity)

Why this should work on BTC/ETH (not just SOL):
- CRSI mean reversion works in bear/range markets (2025 test period)
- HTF trend filter avoids counter-trend trades that failed in #213, #219
- Choppiness avoids whipsaw entries that destroyed #214, #222
- Looser thresholds ensure we don't get 0 trades like #215, #218

TARGET: Sharpe > 0.5 on ALL symbols, 40-70 trades/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on up/down streak duration
    PercentRank(100): Percentile rank of today's return over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on price
    rsi_price = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent Rank (100)
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=range(n), dtype=float)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        if len(window) == rank_period:
            percent_rank.iloc[i] = (returns.iloc[i] > window).sum() / rank_period * 100
    percent_rank = percent_rank.fillna(50.0).values
    
    # CRSI
    crsi = (rsi_price + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    crsi_3_2_100 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate 4h HMA for trend direction (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start later to ensure all indicators ready (CRSI needs 100+)
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi_3_2_100[i]):
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC for liquidity) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF TREND BIAS (4h HMA21) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === MACRO BIAS (1d HMA21) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        choppy_regime = chop_14[i] > 50.0  # Range/mean-revert mode
        trending_regime = chop_14[i] < 40.0  # Trend-follow mode
        neutral_regime = not choppy_regime and not trending_regime
        
        # === CRSI EXTREMES (mean reversion signals) ===
        crsi_oversold = crsi_3_2_100[i] < 25.0  # Long entry
        crsi_overbought = crsi_3_2_100[i] > 75.0  # Short entry
        crsi_extreme = crsi_oversold or crsi_overbought
        
        # === SMA200 FILTER (avoid counter-trend in strong trends) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + HTF trend support + session
        if crsi_oversold and in_session:
            # In choppy regime: mean revert long
            if choppy_regime:
                if price_above_hma_4h or price_above_hma_1d:  # At least one HTF bullish
                    new_signal = POSITION_SIZE_FULL
                elif price_above_sma200:  # Above long-term MA
                    new_signal = POSITION_SIZE_HALF
            
            # In trending regime: only long if HTF bullish
            elif trending_regime:
                if price_above_hma_4h and price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL
                elif price_above_hma_4h:
                    new_signal = POSITION_SIZE_HALF
            
            # Neutral regime: conservative long
            else:
                if price_above_hma_4h and price_above_sma200:
                    new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: CRSI overbought + HTF trend support + session
        elif crsi_overbought and in_session:
            # In choppy regime: mean revert short
            if choppy_regime:
                if price_below_hma_4h or price_below_hma_1d:  # At least one HTF bearish
                    new_signal = -POSITION_SIZE_FULL
                elif price_below_sma200:  # Below long-term MA
                    new_signal = -POSITION_SIZE_HALF
            
            # In trending regime: only short if HTF bearish
            elif trending_regime:
                if price_below_hma_4h and price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL
                elif price_below_hma_4h:
                    new_signal = -POSITION_SIZE_HALF
            
            # Neutral regime: conservative short
            else:
                if price_below_hma_4h and price_below_sma200:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and CRSI not at opposite extreme
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought
                if crsi_3_2_100[i] < 80.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold
                if crsi_3_2_100[i] > 20.0:
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
        
        # === CRSI EXIT (take profit at opposite extreme) ===
        if in_position and position_side > 0 and crsi_overbought:
            new_signal = 0.0  # Take profit long
        
        if in_position and position_side < 0 and crsi_oversold:
            new_signal = 0.0  # Take profit short
        
        # === HTF TREND REVERSAL EXIT ===
        # Exit long if 4h HMA turns bearish
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        # Exit short if 4h HMA turns bullish
        if in_position and position_side < 0 and price_above_hma_4h:
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
                # Position flip
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