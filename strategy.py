#!/usr/bin/env python3
"""
Experiment #188: 30m Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Volume Session

Hypothesis: Previous 30m strategies failed due to either (a) too many trades causing fee drag,
or (b) too strict filters generating 0 trades. This strategy combines:

1. EHLERS FISHER TRANSFORM (period=9): Catches reversals in bear/range markets better than RSI.
   Long when Fisher crosses above -1.5, short when crosses below +1.5.
2. 4h HMA(21) TREND: Direction bias only (don't fight the HTF trend).
3. 1d ADX(14) REGIME: ADX>25 = trend (follow), ADX<20 = range (mean revert).
4. VOLUME CONFIRMATION: volume > 0.8x 20-period average (avoid low-liquidity traps).
5. SESSION FILTER: Only trade 8-20 UTC (reduce overnight noise, ~50% trade reduction).
6. ATR TRAILING STOP: 2.5x ATR(14) to protect capital.

Why this should work on 30m:
- Fisher Transform has superior reversal detection vs RSI in literature
- 4h HMA provides trend bias without over-filtering
- 1d ADX distinguishes when to trend-follow vs mean-revert
- Session + volume filters cut trade frequency to 40-80/year target
- Position size 0.25 (conservative for lower TF)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (max 0.30 for lower TF)
Target trades: 40-80/year per symbol (critical for 30m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_hma_adx_volume_session_v1"
timeframe = "30m"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * ((close - lowest) / (highest - lowest) - 0.5)
    Signals: Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate highest high and lowest low over period
    highest = high_s.rolling(window=period, min_periods=period).max().values
    lowest = low_s.rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # Normalize price position within range
    x = 0.67 * ((close - lowest) / price_range - 0.5)
    x = np.clip(x, -0.99, 0.99)  # Prevent ln domain errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Signal line (previous Fisher value for crossover detection)
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    return fisher, fisher_prev

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

def extract_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) // 3600) % 24
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
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d indicators
    adx_1d, _, _ = calculate_adx(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract session hours
    hours = extract_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 30m)
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
        
        if np.isnan(adx_1d_aligned[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D ADX REGIME ===
        is_trend_regime = adx_1d_aligned[i] > 25
        is_range_regime = adx_1d_aligned[i] < 20
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_down = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_trend_regime and not is_range_regime:
            current_size = BASE_SIZE * 0.7  # Reduce size in unclear regime
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Require 3+ confluence factors
        long_confluence = 0
        
        # Factor 1: Fisher crossover or oversold
        if fisher_cross_up or fisher_oversold:
            long_confluence += 1
        
        # Factor 2: 4h trend bullish OR price above 4h HMA
        if trend_4h_bullish or price_above_4h_hma:
            long_confluence += 1
        
        # Factor 3: Volume confirmation
        if volume_confirmed:
            long_confluence += 1
        
        # Factor 4: Session filter
        if in_session:
            long_confluence += 1
        
        # Factor 5: Range regime (mean revert favorable for longs at bottom)
        if is_range_regime and fisher_oversold:
            long_confluence += 1
        
        # Factor 6: Trend regime with bullish bias (trend follow)
        if is_trend_regime and trend_4h_bullish:
            long_confluence += 1
        
        # Entry threshold: need 3+ confluence factors
        if long_confluence >= 3:
            new_signal = current_size
        elif long_confluence >= 2 and bars_since_last_trade > 100:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confluence = 0
        
        # Factor 1: Fisher crossover or overbought
        if fisher_cross_down or fisher_overbought:
            short_confluence += 1
        
        # Factor 2: 4h trend bearish OR price below 4h HMA
        if trend_4h_bearish or price_below_4h_hma:
            short_confluence += 1
        
        # Factor 3: Volume confirmation
        if volume_confirmed:
            short_confluence += 1
        
        # Factor 4: Session filter
        if in_session:
            short_confluence += 1
        
        # Factor 5: Range regime (mean revert favorable for shorts at top)
        if is_range_regime and fisher_overbought:
            short_confluence += 1
        
        # Factor 6: Trend regime with bearish bias
        if is_trend_regime and trend_4h_bearish:
            short_confluence += 1
        
        if short_confluence >= 3:
            new_signal = -current_size
        elif short_confluence >= 2 and bars_since_last_trade > 100:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~100 hours on 30m = 4+ days)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and fisher[i] < -0.5:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and fisher[i] > 0.5:
                new_signal = -current_size * 0.4
            elif fisher[i] < -1.2:
                new_signal = current_size * 0.3
            elif fisher[i] > 1.2:
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
        
        # === FISHER REVERSAL EXIT ===
        fisher_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] > 1.5:
                fisher_reversal = True
            if position_side < 0 and fisher[i] < -1.5:
                fisher_reversal = True
        
        if stoploss_triggered or fisher_reversal:
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