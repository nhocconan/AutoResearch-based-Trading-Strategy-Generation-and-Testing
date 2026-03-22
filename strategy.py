#!/usr/bin/env python3
"""
Experiment #066: 12h Primary + 1d HTF — Dual Regime Simplified

Hypothesis: Previous 12h strategies failed due to overly complex filters causing 0 trades
or negative Sharpe. This strategy uses a SIMPLIFIED dual-regime approach:

1. REGIME DETECTION: ADX(14) determines market state
   - ADX > 20 = Trending regime (follow 1d HMA direction)
   - ADX <= 20 = Ranging regime (mean revert at BB bounds)

2. TRENDING REGIME (ADX > 20):
   - Long: 1d HMA slope > 0 + 12h price > 12h HMA(21) + RSI(14) > 45
   - Short: 1d HMA slope < 0 + 12h price < 12h HMA(21) + RSI(14) < 55
   - Entry on 12h HMA(8/21) crossover for timing

3. RANGING REGIME (ADX <= 20):
   - Long: Price < BB lower band + RSI(14) < 35
   - Short: Price > BB upper band + RSI(14) > 65
   - Exit when price crosses middle band (SMA20)

4. STOPLOSS: 1.5x ATR(14) trailing (tighter than 2.0x for better DD control)

5. POSITION SIZING: 0.25 discrete (conservative for 12h timeframe)

6. TRADE FREQUENCY SAFEGUARD: If no trades for 120 bars (~60 days), 
   allow weaker entry conditions to ensure minimum trade count

Why this should work:
- Dual regime adapts to crypto's tendency to range 60-70% of time
- Lower ADX threshold (20 vs 25) generates more trades
- Mean reversion in ranges captures crypto's oscillatory behavior
- Simpler logic = fewer conflicting conditions = more trades
- 12h naturally limits to 20-50 trades/year target
- Tighter stoploss reduces drawdown from previous -40% to target -25%

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete
Stoploss: 1.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_simplified_1d_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return sma.values, upper.values, lower.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Bollinger Bands for mean reversion
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    # HMA for trend following
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_8 = calculate_hma(close, 8)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_8[i]):
            continue
        
        # === REGIME DETECTION ===
        # ADX > 20 = Trending, ADX <= 20 = Ranging
        is_trending = adx_14[i] > 20
        is_ranging = adx_14[i] <= 20
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0
        trend_1d_bearish = hma_1d_slope_aligned[i] < 0
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 12H TREND ===
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        
        # HMA crossover signals
        hma_bullish_cross = hma_12h_8[i] > hma_12h_21[i] and hma_12h_8[i-1] <= hma_12h_21[i-1]
        hma_bearish_cross = hma_12h_8[i] < hma_12h_21[i] and hma_12h_8[i-1] >= hma_12h_21[i-1]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # --- TRENDING REGIME (ADX > 20) ---
        if is_trending:
            # LONG: 1d bullish + 12h bullish + RSI confirmation
            if (trend_1d_bullish or price_above_1d_hma):
                if price_above_12h_hma and rsi_14[i] > 45:
                    if hma_bullish_cross:
                        new_signal = current_size
                    elif hma_12h_8[i] > hma_12h_21[i]:
                        new_signal = current_size * 0.8
            
            # SHORT: 1d bearish + 12h bearish + RSI confirmation
            if (trend_1d_bearish or price_below_1d_hma):
                if price_below_12h_hma and rsi_14[i] < 55:
                    if hma_bearish_cross:
                        new_signal = -current_size
                    elif hma_12h_8[i] < hma_12h_21[i]:
                        new_signal = -current_size * 0.8
        
        # --- RANGING REGIME (ADX <= 20) ---
        if is_ranging:
            # LONG: Price at lower BB + RSI oversold
            if close[i] < bb_lower[i] and rsi_14[i] < 35:
                new_signal = current_size * 0.7
            
            # SHORT: Price at upper BB + RSI overbought
            if close[i] > bb_upper[i] and rsi_14[i] > 65:
                new_signal = -current_size * 0.7
            
            # Exit mean reversion when price crosses middle band
            if in_position:
                if position_side > 0 and close[i] > bb_mid[i]:
                    new_signal = 0.0
                if position_side < 0 and close[i] < bb_mid[i]:
                    new_signal = 0.0
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 120 bars (~60 days on 12h), allow weaker entry
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and price_above_12h_hma and rsi_14[i] > 40:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and price_below_12h_hma and rsi_14[i] < 60:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 1.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for longs
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 1.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for shorts
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 1.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend reverses bearish strongly
            if position_side > 0 and trend_1d_bearish and price_below_12h_hma:
                trend_reversal = True
            # Exit short if 1d trend reverses bullish strongly
            if position_side < 0 and trend_1d_bullish and price_above_12h_hma:
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
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same direction, keep position (no update needed)
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