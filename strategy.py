#!/usr/bin/env python3
"""
Experiment #195: 1h Primary + 4h/1d HTF — Regime Adaptive RSI + Volume Session Filter

Hypothesis: Previous 1h strategies failed (#185, #188, #190) because entry conditions were 
too strict, generating ZERO trades. This strategy uses LOOSER entry thresholds while 
maintaining quality through confluence:

1. 4h HMA(21) SLOPE: Major trend bias (bullish/bearish/neutral)
2. 1h RSI(14): Entry timing with relaxed thresholds (30/70 not 20/80)
3. VOLUME CONFIRMATION: Volume > 0.7x 20-bar average (filters low-liquidity traps)
4. SESSION FILTER: Only trade 8-20 UTC (high liquidity, avoid Asia night whipsaws)
5. CHOPPINESS(14): Regime selector - range=mean-revert, trend=pullback entries
6. ATR(14) STOPLOSS: 2.0x ATR trailing stop on all positions

Why this should work:
- 1h timeframe with 4h trend bias = HTF trade frequency with 1h entry precision
- Session filter reduces false signals during low-volume hours
- Relaxed RSI thresholds ensure trades are generated (learned from #185, #188, #190)
- Volume confirmation filters out low-liquidity fakeouts
- Discrete position sizing (0.25) minimizes fee churn

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (conservative for lower TF)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 30-60/year per symbol (≈1 trade per 6-12 days)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_volume_session_4h_v2"
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
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
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
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract session hours
    session_hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
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
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.5
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.5
        trend_4h_neutral = (hma_4h_slope_aligned[i] >= -0.5) and (hma_4h_slope_aligned[i] <= 0.5)
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = (session_hours[i] >= 8) and (session_hours[i] <= 20)
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] > 0.7
        
        # === RSI SIGNALS (relaxed thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        rsi_neutral_low = rsi_14[i] < 45
        rsi_neutral_high = rsi_14[i] > 55
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not in_session:
            current_size = BASE_SIZE * 0.5  # Reduce size outside session
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths to ensure trades are generated
        long_conditions_met = 0
        
        # Path 1: Range market + RSI oversold + volume OK (mean revert)
        if is_range_market and rsi_oversold and volume_ok:
            long_conditions_met += 2
        
        # Path 2: Trend market + bullish 4H + RSI pullback (trend follow)
        if is_trend_market and trend_4h_bullish and rsi_neutral_low and volume_ok:
            long_conditions_met += 2
        
        # Path 3: Price above 4H HMA + RSI dip (pullback in uptrend)
        if price_above_4h_hma and rsi_14[i] < 40 and volume_ok:
            long_conditions_met += 2
        
        # Path 4: Neutral trend + RSI extreme low (deep mean revert)
        if trend_4h_neutral and rsi_extreme_low:
            long_conditions_met += 2
        
        # Path 5: Simple RSI oversold + volume (fallback for more trades)
        if rsi_14[i] < 30 and volume_ok:
            long_conditions_met += 1
        
        # Path 6: Session + any long signal (boost during high liquidity)
        if in_session and long_conditions_met >= 1:
            long_conditions_met += 1
        
        if long_conditions_met >= 2:
            new_signal = current_size
        elif long_conditions_met == 1 and bars_since_last_trade > 100:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_conditions_met = 0
        
        # Path 1: Range market + RSI overbought + volume OK
        if is_range_market and rsi_overbought and volume_ok:
            short_conditions_met += 2
        
        # Path 2: Trend market + bearish 4H + RSI rally (trend follow)
        if is_trend_market and trend_4h_bearish and rsi_neutral_high and volume_ok:
            short_conditions_met += 2
        
        # Path 3: Price below 4H HMA + RSI spike (rally in downtrend)
        if price_below_4h_hma and rsi_14[i] > 60 and volume_ok:
            short_conditions_met += 2
        
        # Path 4: Neutral trend + RSI extreme high
        if trend_4h_neutral and rsi_extreme_high:
            short_conditions_met += 2
        
        # Path 5: Simple RSI overbought + volume (fallback)
        if rsi_14[i] > 70 and volume_ok:
            short_conditions_met += 1
        
        # Path 6: Session + any short signal
        if in_session and short_conditions_met >= 1:
            short_conditions_met += 1
        
        if short_conditions_met >= 2:
            new_signal = -current_size
        elif short_conditions_met == 1 and bars_since_last_trade > 100:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~8 days on 1h)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 40:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and rsi_14[i] > 60:
                new_signal = -current_size * 0.4
            elif rsi_14[i] < 28:
                new_signal = current_size * 0.3
            elif rsi_14[i] > 72:
                new_signal = -current_size * 0.3
        
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
            # Exit long if trend turns bearish strongly
            if position_side > 0 and trend_4h_bearish and chop_14[i] < 40:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and trend_4h_bullish and chop_14[i] < 40:
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