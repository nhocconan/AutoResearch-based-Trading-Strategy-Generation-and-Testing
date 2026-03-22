#!/usr/bin/env python3
"""
Experiment #155: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Session Filter

Hypothesis: Previous 1h strategies failed due to either (1) too many trades → fee drag,
or (2) too strict filters → 0 trades. This strategy combines:

1. EHLERS FISHER TRANSFORM (period=9): Superior reversal detection vs RSI in bear markets.
   Long when Fisher crosses above -1.5, short when crosses below +1.5.
2. 4h HMA(21) TREND: Major direction bias — only long if 4h HMA slope > 0, vice versa.
3. 1d HMA(21) REGIME: Ultra-HTF filter — avoid counter-trend when 1d trend is strong.
4. SESSION FILTER (8-20 UTC): Only trade during high liquidity hours (reduces noise).
5. VOLUME CONFIRMATION: Volume > 0.8x 20-bar avg (avoids low-volume fakeouts).
6. CHOPPINESS INDEX: Regime-aware sizing (reduce size in choppy markets).

Why this should work on 1h:
- Fisher Transform catches reversals better than RSI (proven in literature)
- 4h/1d HTF filters reduce trade frequency to target 30-60/year
- Session filter eliminates Asian session noise (low liquidity whipsaws)
- Volume confirmation prevents fake breakouts
- Asymmetric: more aggressive in trending regimes, conservative in range

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base (smaller for 1h to reduce fee impact)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 30-60/year per symbol (critical for 1h profitability)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_session_4h1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    Signals: cross above -1.5 = long, cross below +1.5 = short
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate median price (HL2)
    median = (high_s + low_s) / 2.0
    
    # Highest high and lowest low over period
    highest = median.rolling(window=period, min_periods=period).max()
    lowest = median.rolling(window=period, min_periods=period).min()
    
    # Normalize price
    price_range = highest - lowest
    price_range = price_range.replace(0, 1e-10)
    
    x = 0.67 * (median - lowest) / price_range - 0.33
    x = np.clip(x, -0.999, 0.999)  # Prevent log domain errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Signal line (1-bar lag)
    fisher_prev = fisher.shift(1)
    
    return fisher.values, fisher_prev.values

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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / np.where(vol_avg > 0, vol_avg, 1e-10)
    return vol_ratio

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract session hours
    session_hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    BASE_SIZE = 0.25
    TREND_SIZE = 0.30
    RANGE_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    # Fisher crossover tracking
    fisher_crossed_up = np.zeros(n, dtype=bool)
    fisher_crossed_down = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if fisher_prev[i] < -1.5 and fisher[i] >= -1.5:
            fisher_crossed_up[i] = True
        if fisher_prev[i] > 1.5 and fisher[i] <= 1.5:
            fisher_crossed_down[i] = True
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 4H TREND BIAS (primary direction filter) ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D REGIME (ultra-HTF filter) ===
        trend_1d_strong_bull = hma_1d_slope_aligned[i] > 0.5
        trend_1d_strong_bear = hma_1d_slope_aligned[i] < -0.5
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (session_hours[i] >= 8) and (session_hours[i] <= 20)
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === FISHER SIGNALS ===
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        fisher_cross_up = fisher_crossed_up[i]
        fisher_cross_down = fisher_crossed_down[i]
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === POSITION SIZING BASED ON REGIME ===
        if is_trend_market:
            current_size = TREND_SIZE
        elif is_range_market:
            current_size = RANGE_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - require 3+ confluence
        long_confluence = 0
        
        # Must be in session (critical for 1h)
        if not in_session:
            long_confluence = -10
        else:
            # Confluence 1: 4h trend bullish or neutral
            if trend_4h_bullish or (not trend_4h_bearish and price_above_4h_hma):
                long_confluence += 1
            
            # Confluence 2: Fisher oversold or cross up
            if fisher_oversold or fisher_cross_up:
                long_confluence += 1
            
            # Confluence 3: RSI confirmation
            if rsi_oversold:
                long_confluence += 1
            
            # Confluence 4: Volume confirmation
            if volume_ok:
                long_confluence += 1
            
            # Confluence 5: Not fighting 1d strong bear
            if not trend_1d_strong_bear:
                long_confluence += 1
        
        # Require 3+ confluence for entry (reduces trade frequency)
        if long_confluence >= 3:
            new_signal = current_size
        elif long_confluence == 2 and bars_since_last_trade > 120:
            # Allow weaker signal if no trade for 5+ days
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confluence = 0
        
        if not in_session:
            short_confluence = -10
        else:
            # Confluence 1: 4h trend bearish or neutral
            if trend_4h_bearish or (not trend_4h_bullish and price_below_4h_hma):
                short_confluence += 1
            
            # Confluence 2: Fisher overbought or cross down
            if fisher_overbought or fisher_cross_down:
                short_confluence += 1
            
            # Confluence 3: RSI confirmation
            if rsi_overbought:
                short_confluence += 1
            
            # Confluence 4: Volume confirmation
            if volume_ok:
                short_confluence += 1
            
            # Confluence 5: Not fighting 1d strong bull
            if not trend_1d_strong_bull:
                short_confluence += 1
        
        if short_confluence >= 3:
            new_signal = -current_size
        elif short_confluence == 2 and bars_since_last_trade > 120:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~8 days on 1h)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and fisher[i] < -0.5:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and fisher[i] > 0.5:
                new_signal = -current_size * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bearish and price_below_4h_hma:
                regime_reversal = True
            if position_side < 0 and trend_4h_bullish and price_above_4h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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