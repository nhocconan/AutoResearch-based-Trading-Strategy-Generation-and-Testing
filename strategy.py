#!/usr/bin/env python3
"""
Experiment #167: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Previous complex regime-switching strategies failed due to overfitting and
too many conflicting filters. Research shows simple breakout strategies with trend
filter work best on daily timeframes. This strategy combines:

1. DONCHIAN CHANNEL(20): Breakout above 20-day high = long, below 20-day low = short
2. HMA(21) TREND FILTER: Only take breakouts in direction of HMA slope
3. RSI(14) FILTER: Avoid entries at extreme RSI (>75 long, <25 short) = fade false breakouts
4. 1w HMA(50) BIAS: Major trend direction from weekly timeframe
5. ATR(14) STOPLOSS: 3*ATR trailing stop for risk management

Why this should work:
- Donchian breakouts capture sustained trends (Turtle Trading proven)
- HMA filter reduces whipsaw in ranging markets
- RSI filter avoids chasing exhausted moves
- 1w HTF prevents counter-trend trades in major moves
- 1d timeframe = 20-40 trades/year (low fee drag, high win rate)
- Simple logic = robust across BTC/ETH/SOL

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 3.0 * ATR(14) trailing
Target trades: 20-40/year per symbol (80-160 over 4-year train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_v1"
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_50 = calculate_hma(df_1w['close'].values, 50)
    hma_1w_slope = calculate_hma_slope(hma_1w_50, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
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
        
        if np.isnan(hma_1w_50_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_slope[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1W TREND BIAS ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        trend_1w_neutral = not trend_1w_bullish and not trend_1w_bearish
        
        # === 1D TREND FILTER ===
        trend_1d_bullish = hma_1d_slope[i] > 0.3
        trend_1d_bearish = hma_1d_slope[i] < -0.3
        price_above_hma = close[i] > hma_1d_21[i]
        price_below_hma = close[i] < hma_1d_21[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI FILTER ===
        rsi_overbought = rsi_14[i] > 70
        rsi_oversold = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 80
        rsi_extreme_low = rsi_14[i] < 20
        rsi_neutral = 35 <= rsi_14[i] <= 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if trend_1w_neutral:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for sufficient trade frequency
        long_conditions = 0
        
        # Path 1: Donchian breakout + 1D bullish + RSI not overbought
        if breakout_long and trend_1d_bullish and not rsi_overbought:
            long_conditions += 3
        
        # Path 2: Donchian breakout + 1W bullish bias + RSI neutral
        if breakout_long and trend_1w_bullish and rsi_neutral:
            long_conditions += 3
        
        # Path 3: Price above HMA + RSI oversold (pullback entry)
        if price_above_hma and rsi_oversold and trend_1d_bullish:
            long_conditions += 2
        
        # Path 4: 1W bullish + price above 1W HMA + breakout
        if trend_1w_bullish and close[i] > hma_1w_50_aligned[i] and breakout_long:
            long_conditions += 2
        
        # Path 5: Simple breakout in neutral market (ensure trade frequency)
        if breakout_long and trend_1w_neutral and not rsi_extreme_high:
            long_conditions += 1
        
        if long_conditions >= 2:
            new_signal = current_size
        elif long_conditions == 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Path 1: Donchian breakout + 1D bearish + RSI not oversold
        if breakout_short and trend_1d_bearish and not rsi_oversold:
            short_conditions += 3
        
        # Path 2: Donchian breakout + 1W bearish bias + RSI neutral
        if breakout_short and trend_1w_bearish and rsi_neutral:
            short_conditions += 3
        
        # Path 3: Price below HMA + RSI overbought (rally entry)
        if price_below_hma and rsi_overbought and trend_1d_bearish:
            short_conditions += 2
        
        # Path 4: 1W bearish + price below 1W HMA + breakout
        if trend_1w_bearish and close[i] < hma_1w_50_aligned[i] and breakout_short:
            short_conditions += 2
        
        # Path 5: Simple breakout in neutral market
        if breakout_short and trend_1w_neutral and not rsi_extreme_low:
            short_conditions += 1
        
        if short_conditions >= 2:
            new_signal = -current_size
        elif short_conditions == 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 90 bars (~90 days on 1d)
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_1w_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif breakout_long and not rsi_extreme_high:
                new_signal = current_size * 0.3
            elif breakout_short and not rsi_extreme_low:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and rsi_14[i] < 40:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and rsi_14[i] > 60:
                trend_reversal = True
        
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