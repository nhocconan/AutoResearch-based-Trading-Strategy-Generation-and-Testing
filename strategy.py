#!/usr/bin/env python3
"""
Experiment #203: 1d Primary + 1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Previous Connors RSI + Choppiness strategies failed due to over-filtering
(0 trades on multiple symbols). Research shows Donchian breakout + HMA trend worked
well on SOL (Sharpe +0.782) and is simpler/more reliable. This strategy uses:

1. DONCHIAN BREAKOUT (20): Price breaks 20-day high/low for entry trigger
2. HMA(21) TREND: 1w HMA slope determines major trend bias
3. RSI(14) FILTER: RSI > 55 for longs, RSI < 45 for shorts (momentum confirmation)
4. ADX(14) REGIME: ADX > 25 = trend (follow breakouts), ADX < 20 = range (avoid)
5. ATR TRAILING STOP: 2.5 * ATR(14) for risk management

Why this should work:
- Donchian breakout is proven trend-following (Turtle Trading)
- 1w HTF prevents fighting major trends (critical for 2022 crash survival)
- 1d timeframe = 20-50 trades/year target (low fee drag)
- Simpler logic = more trades (avoids 0-trade failure mode)
- Asymmetric: only trade breakouts in 1w trend direction

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
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
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-day high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    
    # Donchian breakout detection
    donch_breakout_long = close > np.roll(donch_upper, 1)
    donch_breakout_short = close < np.roll(donch_lower, 1)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    TREND_SIZE = 0.35  # More aggressive in trend direction
    COUNTER_SIZE = 0.20  # Conservative counter-trend
    
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        # === 1W TREND BIAS ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === ADX REGIME ===
        is_trending = adx_14[i] > 22  # Lowered from 25 for more trades
        is_ranging = adx_14[i] < 18
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = donch_breakout_long[i]
        breakout_short = donch_breakout_short[i]
        
        # === POSITION SIZING ===
        bars_since_last_trade = i - last_trade_bar
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRIES - Prioritize trend-aligned breakouts
        if breakout_long:
            # Path 1: Trend-aligned breakout (1w bullish + ADX trending + RSI strong)
            if trend_1w_bullish and is_trending and rsi_strong_bull:
                new_signal = TREND_SIZE
            # Path 2: Breakout with 1w HMA support (price above 1w HMA)
            elif price_above_1w_hma and rsi_bullish:
                new_signal = BASE_SIZE
            # Path 3: Simple breakout in trending market
            elif is_trending and rsi_14[i] > 48:
                new_signal = BASE_SIZE * 0.8
            # Path 4: Breakout after long consolidation (bars since trade > 60)
            elif bars_since_last_trade > 60 and rsi_14[i] > 45:
                new_signal = BASE_SIZE * 0.6
        
        # SHORT ENTRIES
        if breakout_short:
            # Path 1: Trend-aligned breakout (1w bearish + ADX trending + RSI strong)
            if trend_1w_bearish and is_trending and rsi_strong_bear:
                new_signal = -TREND_SIZE
            # Path 2: Breakout with 1w HMA resistance (price below 1w HMA)
            elif price_below_1w_hma and rsi_bearish:
                new_signal = -BASE_SIZE
            # Path 3: Simple breakout in trending market
            elif is_trending and rsi_14[i] < 52:
                new_signal = -BASE_SIZE * 0.8
            # Path 4: Breakout after long consolidation
            elif bars_since_last_trade > 60 and rsi_14[i] < 55:
                new_signal = -BASE_SIZE * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 90 bars (~90 days on 1d)
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and rsi_14[i] > 52:
                new_signal = BASE_SIZE * 0.4
            elif trend_1w_bearish and rsi_14[i] < 48:
                new_signal = -BASE_SIZE * 0.4
            elif breakout_long and rsi_14[i] > 50:
                new_signal = BASE_SIZE * 0.3
            elif breakout_short and rsi_14[i] < 50:
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1w_bearish and price_below_1w_hma:
                trend_reversal = True
            if position_side < 0 and trend_1w_bullish and price_above_1w_hma:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True  # Take profit on overbought
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True  # Take profit on oversold
        
        if stoploss_triggered or trend_reversal or rsi_exit:
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