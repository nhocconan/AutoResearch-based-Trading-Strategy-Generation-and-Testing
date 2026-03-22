#!/usr/bin/env python3
"""
Experiment #173: 1d Primary + 1w HTF — Simplified Regime Mean Reversion

Hypothesis: Previous strategies failed due to overly complex entry conditions
resulting in 0 trades (Sharpe=0.000). Research shows simpler mean reversion
with regime filtering works best for BTC/ETH in bear/range markets.

Key changes from failed experiments:
1. SIMPLER entries: RSI(14) extremes only (not Connors RSI which is too strict)
2. WEEKLY trend filter: HMA(21) on 1w for major direction bias
3. CHOPPINESS regime: CHOP>55 = range (fade extremes), CHOP<45 = trend (pullback entries)
4. ASYMMETRIC sizing: 0.30 in favorable regime, 0.20 in counter-trend
5. GENEROUS thresholds: RSI<35/>65 (not <20/>80) to ensure trade frequency

Why this should work:
- 1d timeframe = 20-50 trades/year target (matches cost model)
- Weekly HMA prevents fighting major trends (2022 crash protection)
- Choppiness filter adapts to market regime (range vs trend)
- Simple RSI extremes = more trades than Connors RSI
- Asymmetric sizing reduces counter-trend risk

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_rsi_hma_1w_v1"
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

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
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
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # 200-day SMA for major trend filter
    close_s = pd.Series(close)
    sma_200 = close_s.rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === RSI EXTREMES (simpler than Connors) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === POSITION SIZING ===
        # Favorable regime = larger size, counter-trend = smaller size
        long_favorable = weekly_bullish or price_above_sma200 or is_range_market
        short_favorable = weekly_bearish or price_below_sma200 or is_range_market
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_entry = False
        
        # Path 1: Range market + RSI oversold (mean revert - highest priority)
        if is_range_market and rsi_oversold:
            long_entry = True
        
        # Path 2: Weekly bullish + RSI pullback (trend follow pullback)
        if weekly_bullish and rsi_14[i] < 45:
            long_entry = True
        
        # Path 3: Price above SMA200 + RSI extreme (bull market dip)
        if price_above_sma200 and rsi_extreme_low:
            long_entry = True
        
        # Path 4: Weekly bullish + price above SMA200 + any RSI<50
        if weekly_bullish and price_above_sma200 and rsi_14[i] < 50:
            long_entry = True
        
        if long_entry:
            size = BASE_SIZE if long_favorable else REDUCED_SIZE
            new_signal = size
        
        # SHORT ENTRIES
        short_entry = False
        
        # Path 1: Range market + RSI overbought (mean revert)
        if is_range_market and rsi_overbought:
            short_entry = True
        
        # Path 2: Weekly bearish + RSI rally (trend follow pullback)
        if weekly_bearish and rsi_14[i] > 55:
            short_entry = True
        
        # Path 3: Price below SMA200 + RSI extreme (bear market rally)
        if price_below_sma200 and rsi_extreme_high:
            short_entry = True
        
        # Path 4: Weekly bearish + price below SMA200 + any RSI>50
        if weekly_bearish and price_below_sma200 and rsi_14[i] > 50:
            short_entry = True
        
        if short_entry:
            size = BASE_SIZE if short_favorable else REDUCED_SIZE
            # Don't enter short if long signal also triggered (conflict)
            if new_signal > 0:
                new_signal = 0.0  # flat on conflict
            else:
                new_signal = -size
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~60 days on 1d)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and rsi_14[i] < 45:
                new_signal = REDUCED_SIZE * 0.8
            elif weekly_bearish and rsi_14[i] > 55:
                new_signal = -REDUCED_SIZE * 0.8
            elif is_range_market and rsi_14[i] < 35:
                new_signal = REDUCED_SIZE * 0.7
            elif is_range_market and rsi_14[i] > 65:
                new_signal = -REDUCED_SIZE * 0.7
        
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
        # Exit if major regime changes against position
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and weekly_bearish and chop_14[i] < 40:
                regime_reversal = True
            if position_side < 0 and weekly_bullish and chop_14[i] < 40:
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