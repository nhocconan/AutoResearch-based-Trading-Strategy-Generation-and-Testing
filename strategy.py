#!/usr/bin/env python3
"""
Experiment #036: 12h Primary + 1d HTF — Simplified Regime Adaptive

Hypothesis: Previous strategies failed due to overly strict entry conditions (0 trades).
This strategy uses SIMPLIFIED confluence to ensure 20-50 trades/year:

1. 1d HMA(21) for MAJOR trend bias (price vs HMA determines long/short preference)
2. 12h RSI(14) for entry timing (35-65 for trend, <30 or >70 for reversal)
3. Choppiness Index(14) for regime (choppy = mean revert, trending = follow)
4. ATR(14) for volatility scaling and stoploss (2.5x)
5. Volume filter (simple > 0.7x avg)
6. Discrete sizing (0.25/0.30) with trailing stop

Why this should work:
- Fewer filters = more trades (avoid 0-trade failure mode)
- 12h timeframe = naturally 20-50 trades/year
- Regime-adaptive = works in both bull and bear markets
- Proper stoploss = controls drawdown
- HTF trend filter = avoids counter-trend trades

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_rsi_hma_1d_v1"
timeframe = "12h"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    hh_ll = np.maximum(hh_ll, 1e-10)
    
    chop = 100 * (atr1_sum / atr_period) / hh_ll * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    volume_sma = calculate_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d HMA = bullish bias (prefer longs)
        # Price below 1d HMA = bearish bias (prefer shorts)
        price_vs_hma = close[i] / hma_1d_21_aligned[i]
        trend_1d_bullish = price_vs_hma > 1.0
        trend_1d_bearish = price_vs_hma < 1.0
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean reversion preferred)
        # CHOP < 45 = trend (trend following preferred)
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        
        # === VOLUME FILTER (loose) ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === RSI CONDITIONS (loosened for more trades) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_oversold = rsi_14[i] < 25
        rsi_extreme_overbought = rsi_14[i] > 75
        rsi_neutral = 35 <= rsi_14[i] <= 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED for more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Regime 1: Trending + bullish 1d + RSI pullback (35-50)
        if is_trend and trend_1d_bullish and 35 <= rsi_14[i] <= 50 and volume_ok:
            new_signal = current_size
        
        # Regime 2: Range + extreme oversold RSI (mean reversion)
        elif is_range and rsi_extreme_oversold and volume_ok:
            new_signal = current_size
        
        # Regime 3: Bullish 1d + RSI crossing above 40 (momentum)
        elif trend_1d_bullish and rsi_14[i] > 40 and rsi_14[i-1] <= 40 and volume_ok:
            new_signal = current_size
        
        # SHORT ENTRIES
        # Regime 1: Trending + bearish 1d + RSI pullback (50-65)
        if is_trend and trend_1d_bearish and 50 <= rsi_14[i] <= 65 and volume_ok:
            new_signal = -current_size
        
        # Regime 2: Range + extreme overbought RSI (mean reversion)
        elif is_range and rsi_extreme_overbought and volume_ok:
            new_signal = -current_size
        
        # Regime 3: Bearish 1d + RSI crossing below 60 (momentum)
        elif trend_1d_bearish and rsi_14[i] < 60 and rsi_14[i-1] >= 60 and volume_ok:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # If no trades for 300 bars (~15 days on 12h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                highest_price = max(highest_price, close[i])
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and rsi_14[i] > 70:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and rsi_14[i] < 30:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same direction, maintain position
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