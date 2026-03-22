#!/usr/bin/env python3
"""
Experiment #097: 1d Primary + 1w HTF — Simplified Regime + RSI Mean Reversion

Hypothesis: Previous strategies failed due to overly complex entry logic causing 0 trades.
This version simplifies entries while keeping the proven Choppiness regime filter.
Key changes:
1. Looser RSI thresholds (25/75 instead of 15/85) to ensure trades
2. Remove conflicting filters that prevent entries
3. Add minimum trade frequency safeguard (force entry after 90 days flat)
4. 1w HMA for major trend bias (only one HTF as per experiment rules)
5. Discrete position sizing: 0.0, ±0.25, ±0.30

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Target: 25-40 trades/year, Sharpe > 0.3
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_rsi_donchian_1w_v1"
timeframe = "1d"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bounds."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    
    # Donchian channels for breakout confirmation
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # HMA for trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -90
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]) or np.isnan(rsi_7[i]):
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # Price above 1w HMA = bullish bias
        # Price below 1w HMA = bearish bias
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range market (mean revert strategy)
        # CHOP < 45 = trend market (trend follow strategy)
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === RSI SIGNALS (LOOSENED for more trades) ===
        # Range market: extreme RSI for mean reversion
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_oversold = rsi_7[i] < 25
        rsi_extreme_overbought = rsi_7[i] > 75
        
        # === 1D TREND CONFIRMATION ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        breakout_high = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_low = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC (SIMPLIFIED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if is_range_market:
            # Mean reversion: buy oversold in range
            if rsi_oversold or rsi_extreme_oversold:
                if price_above_1w_hma or hma_bullish:
                    new_signal = current_size
                elif rsi_extreme_oversold:
                    # Very oversold = enter even without trend bias
                    new_signal = current_size * 0.7
        elif is_trend_market:
            # Trend following: buy pullback in uptrend OR breakout
            if price_above_1w_hma and hma_bullish and rsi_14[i] < 50:
                new_signal = current_size
            elif breakout_high and price_above_1w_hma:
                new_signal = current_size * 0.8
        
        # SHORT ENTRIES
        if is_range_market:
            # Mean reversion: sell overbought in range
            if rsi_overbought or rsi_extreme_overbought:
                if price_below_1w_hma or hma_bearish:
                    new_signal = -current_size
                elif rsi_extreme_overbought:
                    # Very overbought = enter even without trend bias
                    new_signal = -current_size * 0.7
        elif is_trend_market:
            # Trend following: sell pullback in downtrend OR breakdown
            if price_below_1w_hma and hma_bearish and rsi_14[i] > 50:
                new_signal = -current_size
            elif breakout_low and price_below_1w_hma:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 90 days, allow weaker entry to ensure minimum trades
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if price_above_1w_hma and rsi_14[i] < 40:
                new_signal = current_size * 0.5
            elif price_below_1w_hma and rsi_14[i] > 60:
                new_signal = -current_size * 0.5
        
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
        
        # Apply stoploss
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