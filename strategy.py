#!/usr/bin/env python3
"""
Experiment #150: 1h Primary + 4h/12h HTF — Fisher Transform + KAMA Trend + Session Filter

Hypothesis: Previous 1h strategies failed due to either too many trades (fee drag) or 
too few trades (0 Sharpe). This strategy combines:

1. EHLERS FISHER TRANSFORM: Better reversal detection than RSI, catches turns in bear rallies
2. KAMA (Kaufman Adaptive): Smooth in trends, responsive in ranges - adapts to volatility
3. 4h HMA(21): Major trend bias - only trade with HTF trend
4. 12h CHOPPINESS: Regime filter - range vs trend markets
5. SESSION FILTER: Only 8-20 UTC (high liquidity, less noise)
6. VOLUME CONFIRMATION: Volume > 0.8x 20-bar average

Why this should work:
- Fisher Transform has superior reversal detection in academic literature
- KAMA adapts to market conditions automatically
- 4h HTF prevents counter-trend trades in strong moves
- Session filter reduces whipsaws during Asian session (low liquidity)
- Volume filter confirms genuine moves vs fakeouts

Timeframe: 1h (REQUIRED)
HTF: 4h + 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (conservative for 1h)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_kama_session_4h12h_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((close - LL) / (HH - LL) - 0.5) + 0.67 * prev_Fisher
    """
    n = len(close)
    fisher = np.zeros(n)
    signal_line = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh == ll:
            fisher[i] = fisher[i-1] if i > 0 else 0
            signal_line[i] = fisher[i-1] if i > 0 else 0
            continue
        
        x = 0.66 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * fisher[i-1]
        x = np.clip(x, -0.999, 0.999)  # Prevent ln domain error
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        signal_line[i] = fisher[i-1]
    
    return fisher, signal_line

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast SC - slow SC) + slow SC)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(er_period, n):
        # Efficiency Ratio
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if noise == 0:
            er = 1.0
        else:
            er = signal / noise
        
        # Smoothing Constant
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market
    CHOP < 38.2 = trend market
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

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Calculate 12h Choppiness
    chop_12h = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    kama_1h = calculate_kama(close, 10, 2, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25  # Conservative for 1h
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100  # Minimum 100 bars between trades
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        if np.isnan(chop_12h_aligned[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(kama_1h[i]) or np.isnan(vol_avg[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === 4H TREND BIAS ===
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_aligned[i]
        
        # === 12H REGIME ===
        is_range_market = chop_12h_aligned[i] > 55
        is_trend_market = chop_12h_aligned[i] < 45
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1]
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1]
        
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        fisher_extreme_low = fisher[i] < -1.5
        fisher_extreme_high = fisher[i] > 1.5
        
        # === KAMA TREND ===
        kama_slope_up = kama_1h[i] > kama_1h[i-5] if i >= 5 else False
        kama_slope_down = kama_1h[i] < kama_1h[i-5] if i >= 5 else False
        
        price_above_kama = close[i] > kama_1h[i]
        price_below_kama = close[i] < kama_1h[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Need 3+ confluence
        long_confluence = 0
        
        # 1. Fisher reversal signal
        if fisher_cross_up or fisher_extreme_low:
            long_confluence += 1
        
        # 2. 4h trend alignment (price above 4h HMA)
        if price_above_4h_hma:
            long_confluence += 1
        
        # 3. KAMA confirmation
        if price_above_kama and kama_slope_up:
            long_confluence += 1
        
        # 4. Volume confirmation
        if volume_confirmed:
            long_confluence += 1
        
        # 5. Session filter
        if in_session:
            long_confluence += 0.5  # Half weight
        
        # 6. Range market mean reversion (add extra if oversold in range)
        if is_range_market and fisher_oversold:
            long_confluence += 1
        
        if long_confluence >= 3.0:
            new_signal = current_size
        elif long_confluence >= 2.5 and bars_since_last_trade > 150:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_confluence = 0
        
        # 1. Fisher reversal signal
        if fisher_cross_down or fisher_extreme_high:
            short_confluence += 1
        
        # 2. 4h trend alignment (price below 4h HMA)
        if price_below_4h_hma:
            short_confluence += 1
        
        # 3. KAMA confirmation
        if price_below_kama and kama_slope_down:
            short_confluence += 1
        
        # 4. Volume confirmation
        if volume_confirmed:
            short_confluence += 1
        
        # 5. Session filter
        if in_session:
            short_confluence += 0.5
        
        # 6. Range market mean reversion
        if is_range_market and fisher_overbought:
            short_confluence += 1
        
        if short_confluence >= 3.0:
            new_signal = -current_size
        elif short_confluence >= 2.5 and bars_since_last_trade > 150:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~8 days on 1h)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if price_above_4h_hma and fisher[i] < -0.5:
                new_signal = current_size * 0.4
            elif price_below_4h_hma and fisher[i] > 0.5:
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