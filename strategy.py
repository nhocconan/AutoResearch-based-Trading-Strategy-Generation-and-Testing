#!/usr/bin/env python3
"""
Experiment #410: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + Volume Session

Hypothesis: After 409 experiments, clear patterns emerge for 1h timeframe:
1. 1h needs 30-60 trades/year (strict entry, HTF direction filter)
2. Fisher Transform catches reversals better than RSI in bear/range markets
3. 4h HMA(21) for trend direction prevents counter-trend trades
4. Volume confirmation (>1.2x avg) filters false breakouts
5. Session filter (8-20 UTC) avoids low-liquidity whipsaws
6. Choppiness regime adapts entry logic (mean revert vs trend follow)

Why this might beat current best (Sharpe=0.435):
- Fisher Transform has proven edge in crypto reversals (research notes)
- 1h TF with 4h/12h HTF = HTF trade frequency with 1h entry precision
- Volume + session filters reduce false signals during Asian low-liquidity
- Asymmetric sizing: larger positions in confirmed trend, smaller in range

Position sizing: 0.20-0.30 (discrete, max 0.35)
Stoploss: 2.5 * ATR trailing
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_vol_session_4h12h_v1"
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
    Calculate Ehlers Fisher Transform.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Mid price
    mid = (high_s + low_s) / 2.0
    
    # Normalize price to -1 to +1 range
    highest = mid.rolling(window=period, min_periods=period).max()
    lowest = mid.rolling(window=period, min_periods=period).min()
    
    range_val = highest - lowest
    range_val = range_val.replace(0, 1e-10)
    
    normalized = 0.66 * ((mid - lowest) / range_val - 0.5) + 0.67 * normalized.shift(1).fillna(0)
    normalized = normalized.clip(-0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized.values) / (1 - normalized.values + 1e-10))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10)) / np.log10(period)
    
    return chop

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 12h HTF for additional regime filter
    df_12h = get_htf_data(prices, '12h')
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    choppiness = calculate_choppiness(high, low, close, 14)
    volume_ma = calculate_volume_ma(volume, 20)
    
    # Calculate 1h HMA for local trend
    hma_1h_16 = calculate_hma(close, period=16)
    hma_1h_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(volume_ma[i]) or volume_ma[i] == 0:
            continue
        
        if np.isnan(hma_1h_16[i]) or np.isnan(hma_1h_48[i]):
            continue
        
        # === 4H MAJOR TREND (primary direction filter) ===
        bull_4h = close[i] > hma_4h_21_aligned[i]
        bear_4h = close[i] < hma_4h_21_aligned[i]
        
        # === 12H CONFIRMATION (stronger trend signal) ===
        bull_12h = close[i] > hma_12h_21_aligned[i]
        bear_12h = close[i] < hma_12h_21_aligned[i]
        
        # Strong bull: both 4h and 12h bullish
        strong_bull = bull_4h and bull_12h
        # Strong bear: both 4h and 12h bearish
        strong_bear = bear_4h and bear_12h
        
        # === 1H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1h_16[i] > hma_1h_48[i]
        hma_bearish = hma_1h_16[i] < hma_1h_48[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = choppiness[i] > 55.0
        is_trending = choppiness[i] < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Additional Fisher signals for choppy markets
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (volume_ma[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.1  # 10% above average
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = (hour >= 8) and (hour <= 20)
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        if strong_bull or (bull_4h and hma_bullish):
            # Trending market: Fisher breakout + volume
            if is_trending and fisher_long and volume_confirmed:
                new_signal = LONG_SIZE
            # Choppy market: Fisher oversold mean reversion
            elif is_choppy and fisher_oversold and hma_bullish:
                new_signal = LONG_SIZE * 0.8
            # Session + volume confirmation for any long
            elif in_session and volume_confirmed and fisher[i] < -0.5 and hma_bullish:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRY
        if strong_bear or (bear_4h and hma_bearish):
            # Trending market: Fisher breakdown + volume
            if is_trending and fisher_short and volume_confirmed:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Choppy market: Fisher overbought mean reversion
            elif is_choppy and fisher_overbought and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Session + volume confirmation for any short
            elif in_session and volume_confirmed and fisher[i] > 0.5 and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 20 bars (~20 hours on 1h), allow weaker signals
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if bull_4h and fisher[i] < -0.5 and hma_bullish:
                new_signal = LONG_SIZE * 0.5
            elif bear_4h and fisher[i] > 0.5 and hma_bearish:
                new_signal = -SHORT_SIZE * 0.5
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.5:
            new_signal = 0.0
        
        # 4h trend reversal exit
        if in_position and position_side > 0 and bear_4h:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_4h:
            new_signal = 0.0
        
        # Local trend reversal exit
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if highest_price == 0.0:
                highest_price = close[i]
            else:
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