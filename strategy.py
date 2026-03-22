#!/usr/bin/env python3
"""
Experiment #385: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime

Hypothesis: After 350+ experiments, 1h strategies keep failing due to:
1. Too many trades → fee drag kills profit (>200/year)
2. Wrong regime logic (trend-follow fails in 2022 crash + 2025 bear)
3. No session/volume filters → noise entries

This strategy uses:
1. 4h HMA(21) for trend direction (proven in best strategies)
2. 1d HMA(21) for major regime filter (bull/bear bias)
3. Fisher Transform(9) for reversal entries (catches bear market rallies)
4. Choppiness Index(14) regime filter (CHOP>55=range, CHOP<45=trend)
5. Session filter (8-20 UTC only) → reduces low-volume noise
6. Volume filter (>0.8x 20-bar avg) → confirms moves
7. Discrete position sizing (0.25 max) → controls drawdown
8. ATR 2.5x trailing stop → limits losses

Why this might beat current best (Sharpe=0.435):
- Fisher Transform works in bear/range markets (2025 test period)
- Choppiness filter avoids trend-follow whipsaws in choppy periods
- 4h/1d HTF gives direction, 1h only for timing → optimal trade freq
- Session filter removes Asian session noise (low volume whipsaws)
- Target: 40-70 trades/year on 1h (within fee drag limits)

Position sizing: 0.25 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_4h1d_session_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear/range markets.
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.66 * ((price - lowest) / (highest - lowest) - 0.5) + 0.67 * prev_X
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    # Calculate typical price
    price = (high + low) / 2.0
    
    x_prev = 0.0
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            x = x_prev
        else:
            x = 0.66 * ((price[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * x_prev
            x = np.clip(x, -0.999, 0.999)  # Prevent division by zero
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
        fisher_signal[i] = fisher[i-1] if i > 0 else 0.0
        
        x_prev = x
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - regime detection.
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * log10(sum(ATR) / (highest_high - lowest_low)) / log10(period)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar (true range)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Calculate 1d HTF indicators (major regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    vol_avg = calculate_volume_avg(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i] if not np.isnan(vol_avg[i]) else True
        
        # === 1D MAJOR REGIME (primary direction bias) ===
        # Price above 1d HMA = bull market (favor longs)
        # Price below 1d HMA = bear market (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND DIRECTION ===
        # Price above 4h HMA = uptrend on 4h
        # Price below 4h HMA = downtrend on 4h
        trend_4h_up = close[i] > hma_4h_21_aligned[i]
        trend_4h_down = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range/choppy (use mean reversion)
        # CHOP < 45 = trending (use trend follow)
        choppy_regime = chop[i] > 55.0
        trending_regime = chop[i] < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Fisher extreme levels for stronger signals
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (3+ confluence required)
        if in_session and volume_ok:
            # Regime 1: Trending + trend follow
            if trending_regime and bull_regime and trend_4h_up:
                if fisher_long and rsi_oversold:
                    new_signal = LONG_SIZE
                elif fisher_oversold and rsi_14[i] < 35:
                    new_signal = LONG_SIZE
            
            # Regime 2: Choppy + mean reversion
            elif choppy_regime and fisher_oversold:
                if rsi_14[i] < 30:
                    new_signal = LONG_SIZE
                elif rsi_14[i] < 40 and bull_regime:
                    new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (3+ confluence required)
        if in_session and volume_ok:
            # Regime 1: Trending + trend follow
            if trending_regime and bear_regime and trend_4h_down:
                if fisher_short and rsi_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
                elif fisher_overbought and rsi_14[i] > 65:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
            
            # Regime 2: Choppy + mean reversion
            elif choppy_regime and fisher_overbought:
                if rsi_14[i] > 70:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
                elif rsi_14[i] > 60 and bear_regime:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 30 bars (~30 hours on 1h), allow weaker entry
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if bull_regime and fisher[i] < -1.0 and rsi_14[i] < 45:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and fisher[i] > 1.0 and rsi_14[i] > 55:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit on reversal)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.5:
            new_signal = 0.0
        
        # RSI extreme exit
        if in_position and position_side > 0 and rsi_14[i] > 70:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30:
            new_signal = 0.0
        
        # Regime flip exit (1d major regime change)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # 4h trend reversal exit
        if in_position and position_side > 0 and trend_4h_down:
            new_signal = 0.0
        if in_position and position_side < 0 and trend_4h_up:
            new_signal = 0.0
        
        # Session exit (close position outside trading hours)
        if in_position and not in_session:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals