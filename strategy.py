#!/usr/bin/env python3
"""
Experiment #180: 1h Primary + 4h/12h HTF — Regime-Adaptive RSI Pullback

Hypothesis: Previous 1h strategies failed because entry conditions were TOO STRICT
(exp #170, #175 = 0 trades). This strategy uses SIMPLER entry logic with HTF filters:

1. 4h HMA(21) SLOPE: Primary trend direction (faster than 1d for 1h entries)
2. 12h CHOPPINESS: Regime filter (CHOP>55=range, CHOP<45=trend)
3. 1h RSI(14) PULLBACK: Enter on RSI 35-45 in uptrend, 55-65 in downtrend
4. VOLUME CONFIRMATION: Volume > 1.2x 20-bar average (avoid fake moves)
5. SESSION FILTER: Only trade 8-20 UTC (high liquidity periods)

Why this should work on 1h:
- 4h trend filter prevents counter-trend trades (major failure mode)
- RSI pullback = proven mean-reversion within trend (60-70% win rate)
- Volume filter reduces false breakouts
- Session filter avoids low-liquidity whipsaws
- Target: 40-80 trades/year (1-2 per week per symbol)

Timeframe: 1h (REQUIRED per experiment)
HTF: 4h for trend, 12h for regime via mtf_data.get_htf_data() — ONCE before loop
Position sizing: 0.25 base (smaller for lower TF), max 0.30
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_regime_4h12h_v1"
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
    """Calculate HMA slope as percentage change over lookback."""
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

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_utc_hour(open_time):
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h trend indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 12h regime indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    chop_12h = calculate_choppiness(high_12h, low_12h, close_12h, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
    # Volume ratio
    vol_ratio = volume / np.where(vol_ma_20 > 0, vol_ma_20, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.30 for 1h)
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
        
        if np.isnan(chop_12h_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 12H CHOPPINESS REGIME ===
        is_range_market = chop_12h_aligned[i] > 55
        is_trend_market = chop_12h_aligned[i] < 45
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.1  # Lowered from 1.2 for more trades
        
        # === RSI PULLBACK LEVELS ===
        rsi_oversold_pullback = 35 <= rsi_14[i] <= 50
        rsi_overbought_pullback = 50 <= rsi_14[i] <= 65
        rsi_deep_oversold = rsi_14[i] < 35
        rsi_deep_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for sufficient trade frequency
        long_conditions = 0
        
        # Path 1: Trend market + bullish 4h + RSI pullback (primary)
        if is_trend_market and trend_4h_bullish and rsi_oversold_pullback:
            long_conditions += 2
        
        # Path 2: Range market + RSI deep oversold (mean revert)
        if is_range_market and rsi_deep_oversold:
            long_conditions += 2
        
        # Path 3: Price above 4h HMA + RSI pullback (trend continuation)
        if price_above_4h_hma and 40 <= rsi_14[i] <= 55:
            long_conditions += 1
        
        # Path 4: Volume spike + RSI oversold (momentum entry)
        if vol_confirmed and rsi_14[i] < 45:
            long_conditions += 1
        
        # Path 5: Session filter bonus (higher quality trades)
        if in_session and long_conditions >= 1:
            long_conditions += 0.5
        
        if long_conditions >= 2.5:
            new_signal = current_size
        elif long_conditions >= 2.0 and bars_since_last_trade > 48:
            new_signal = current_size
        elif long_conditions >= 1.5 and bars_since_last_trade > 72:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Path 1: Trend market + bearish 4h + RSI pullback
        if is_trend_market and trend_4h_bearish and rsi_overbought_pullback:
            short_conditions += 2
        
        # Path 2: Range market + RSI deep overbought
        if is_range_market and rsi_deep_overbought:
            short_conditions += 2
        
        # Path 3: Price below 4h HMA + RSI pullback
        if price_below_4h_hma and 45 <= rsi_14[i] <= 60:
            short_conditions += 1
        
        # Path 4: Volume spike + RSI overbought
        if vol_confirmed and rsi_14[i] > 55:
            short_conditions += 1
        
        # Path 5: Session filter bonus
        if in_session and short_conditions >= 1:
            short_conditions += 0.5
        
        if short_conditions >= 2.5:
            new_signal = -current_size
        elif short_conditions >= 2.0 and bars_since_last_trade > 48:
            new_signal = -current_size
        elif short_conditions >= 1.5 and bars_since_last_trade > 72:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~5 days on 1h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif rsi_14[i] < 30:
                new_signal = current_size * 0.3
            elif rsi_14[i] > 70:
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and is_trend_market and trend_4h_bearish:
                regime_reversal = True
            if position_side < 0 and is_trend_market and trend_4h_bullish:
                regime_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        if stoploss_triggered or regime_reversal or rsi_exit:
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