#!/usr/bin/env python3
"""
Experiment #154: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: Previous 4h strategies failed due to overly strict entry conditions (0 trades)
or fighting the major trend. Research shows HMA trend + RSI pullback works best on 4h
(SOL Sharpe +0.879). This strategy combines:

1. 4h HMA(21/50) crossover for primary trend direction
2. 12h HMA(21) slope for major trend bias (avoid counter-trend trades)
3. RSI(14) pullback entries in trend direction (RSI 35-45 for longs, 55-65 for shorts)
4. Choppiness Index regime switch: CHOP>55 = mean revert at BB bounds, CHOP<45 = trend follow
5. Donchian(20) breakout confirmation for trend entries
6. ATR(14) trailing stoploss at 2.5x ATR

Why this should work:
- 4h timeframe = 25-50 trades/year target (low fee drag, proven TF)
- HMA is faster than EMA, catches trends earlier with less lag
- RSI pullback = enter on dips in uptrend, rallies in downtrend (high win rate)
- 12h HTF prevents fighting major trend (critical for BTC/ETH in bear markets)
- Choppiness filter adapts to market regime (mean revert in chop, trend in trends)
- Loose entry conditions ensure trades generate (lesson from 142 failed strategies)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_donchian_12h_v1"
timeframe = "4h"
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

def calculate_hma_slope(hma_values, lookback=3):
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 4h indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # HMA crossover signal
    hma_bullish = hma_21 > hma_50
    hma_bearish = hma_21 < hma_50
    
    # 12h trend bias
    trend_12h_bullish = hma_12h_slope_aligned > 0.2
    trend_12h_bearish = hma_12h_slope_aligned < -0.2
    
    # Choppiness regime
    is_range_market = chop_14 > 55
    is_trend_market = chop_14 < 45
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for more trades
        long_conditions = 0
        
        # Path 1: HMA bullish + RSI pullback (primary trend follow)
        if hma_bullish[i] and 35 <= rsi_14[i] <= 50:
            long_conditions += 2
        
        # Path 2: 12h bullish bias + RSI dip
        if trend_12h_bullish and rsi_14[i] < 45:
            long_conditions += 2
        
        # Path 3: Range market + BB lower + RSI oversold (mean revert)
        if is_range_market and close[i] < bb_lower[i] and rsi_14[i] < 35:
            long_conditions += 2
        
        # Path 4: Donchian breakout + HMA bullish (momentum)
        if hma_bullish[i] and close[i] > donchian_upper[i] * 0.995:
            long_conditions += 2
        
        # Path 5: RSI very oversold (capitulation long)
        if rsi_14[i] < 25:
            long_conditions += 1
        
        # Path 6: HMA crossover just happened (fresh trend)
        if i > 100 and hma_bullish[i] and not hma_bullish[i-1]:
            long_conditions += 2
        
        if long_conditions >= 3:
            new_signal = BASE_SIZE
        elif long_conditions == 2 and bars_since_last_trade > 40:
            new_signal = BASE_SIZE * 0.6
        elif long_conditions >= 1 and bars_since_last_trade > 80:
            new_signal = BASE_SIZE * 0.4
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Path 1: HMA bearish + RSI rally (primary trend follow)
        if hma_bearish[i] and 50 <= rsi_14[i] <= 65:
            short_conditions += 2
        
        # Path 2: 12h bearish bias + RSI rise
        if trend_12h_bearish and rsi_14[i] > 55:
            short_conditions += 2
        
        # Path 3: Range market + BB upper + RSI overbought (mean revert)
        if is_range_market and close[i] > bb_upper[i] and rsi_14[i] > 65:
            short_conditions += 2
        
        # Path 4: Donchian breakdown + HMA bearish (momentum)
        if hma_bearish[i] and close[i] < donchian_lower[i] * 1.005:
            short_conditions += 2
        
        # Path 5: RSI very overbought (capitulation short)
        if rsi_14[i] > 75:
            short_conditions += 1
        
        # Path 6: HMA crossover just happened (fresh downtrend)
        if i > 100 and hma_bearish[i] and not hma_bearish[i-1]:
            short_conditions += 2
        
        if short_conditions >= 3:
            new_signal = -BASE_SIZE
        elif short_conditions == 2 and bars_since_last_trade > 40:
            new_signal = -BASE_SIZE * 0.6
        elif short_conditions >= 1 and bars_since_last_trade > 80:
            new_signal = -BASE_SIZE * 0.4
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and rsi_14[i] < 40:
                new_signal = BASE_SIZE * 0.4
            elif trend_12h_bearish and rsi_14[i] > 60:
                new_signal = -BASE_SIZE * 0.4
            elif rsi_14[i] < 30:
                new_signal = BASE_SIZE * 0.3
            elif rsi_14[i] > 70:
                new_signal = -BASE_SIZE * 0.3
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if trend reverses strongly
            if position_side > 0 and hma_bearish[i] and rsi_14[i] < 40:
                regime_reversal = True
            # Exit short if trend reverses strongly
            if position_side < 0 and hma_bullish[i] and rsi_14[i] > 60:
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