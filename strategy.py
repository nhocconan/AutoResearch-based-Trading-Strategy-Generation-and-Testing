#!/usr/bin/env python3
"""
Experiment #140: 1h Primary + 4h/12h HTF — Regime-Adaptive Mean Reversion with Confluence

Hypothesis: 1h strategies fail due to either (a) too many trades → fee drag, or (b) too strict → 0 trades.
Solution: Use 4h/12h for trend direction, 1h only for entry timing within HTF trend.
Combine Choppiness Index regime + Connors RSI + HTF HMA trend + relaxed session filter.

Key innovations:
1. 4h HMA(21) determines long/short bias (only trade WITH HTF trend)
2. 12h HMA slope confirms major trend direction
3. Choppiness Index: >55 = range (mean revert), <45 = trend (pullback entries)
4. Connors RSI extremes for entry timing (CRSI<25 long, >75 short)
5. Volume filter: volume > 0.7x 20-bar average (relaxed for more trades)
6. Stoploss: 2.5 * ATR(14) trailing

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for lower TF to reduce fee impact)
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_connors_hma_4h12h_v2"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    BASE_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(vol_avg_20[i]):
            continue
        
        # === SESSION FILTER (relaxed: 6-22 UTC for more trades) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 6 <= hour <= 22
        
        # === VOLUME FILTER (relaxed for more trades) ===
        vol_filter = volume[i] > 0.7 * vol_avg_20[i]
        
        # === 4H TREND BIAS ===
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.15
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.15
        
        # === 12H MAJOR TREND ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.2
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.2
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 50
        is_trend_market = chop_14[i] < 45
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 30
        crsi_overbought = crsi[i] > 70
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - require 2+ confluence (relaxed for more trades)
        long_confluence = 0
        
        # Confluence 1: 4h trend bullish OR range market
        if trend_4h_bullish or is_range_market:
            long_confluence += 1
        
        # Confluence 2: 12h trend not strongly bearish
        if not trend_12h_bearish:
            long_confluence += 1
        
        # Confluence 3: CRSI oversold
        if crsi_oversold:
            long_confluence += 1
        
        # Confluence 4: Price below BB lower (extreme)
        if price_below_bb_lower:
            long_confluence += 1
        
        # Confluence 5: Session + volume
        if in_session and vol_filter:
            long_confluence += 0.5
        
        # Confluence 6: Price above 4h HMA (trend confirmation)
        if price_above_4h_hma:
            long_confluence += 0.5
        
        if long_confluence >= 2.5:
            new_signal = current_size
        elif long_confluence >= 2.0 and bars_since_last_trade > 80:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confluence = 0
        
        # Confluence 1: 4h trend bearish OR range market
        if trend_4h_bearish or is_range_market:
            short_confluence += 1
        
        # Confluence 2: 12h trend not strongly bullish
        if not trend_12h_bullish:
            short_confluence += 1
        
        # Confluence 3: CRSI overbought
        if crsi_overbought:
            short_confluence += 1
        
        # Confluence 4: Price above BB upper (extreme)
        if price_above_bb_upper:
            short_confluence += 1
        
        # Confluence 5: Session + volume
        if in_session and vol_filter:
            short_confluence += 0.5
        
        # Confluence 6: Price below 4h HMA (trend confirmation)
        if price_below_4h_hma:
            short_confluence += 0.5
        
        if short_confluence >= 2.5:
            new_signal = -current_size
        elif short_confluence >= 2.0 and bars_since_last_trade > 80:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 180 bars (~7.5 days on 1h)
        if bars_since_last_trade > 180 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and crsi[i] < 40:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and crsi[i] > 60:
                new_signal = -current_size * 0.4
            elif crsi[i] < 25:
                new_signal = current_size * 0.3
            elif crsi[i] > 75:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
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