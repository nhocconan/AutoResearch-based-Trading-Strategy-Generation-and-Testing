#!/usr/bin/env python3
"""
Experiment #255: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + HTF HMA Trend

Hypothesis: After 254 experiments, the winning pattern combines:
1. 1d HMA(21) for PRIMARY trend direction (proven on #251)
2. 4h Choppiness(14) for regime detection (trend vs mean-revert mode)
3. 1h Fisher Transform(9) for precise entry timing (catches reversals in bear rallies)
4. 1h ATR(14) for volatility filter and stoploss
5. Volume confirmation to avoid low-liquidity traps

Key insight from research: Fisher Transform excels in bear/range markets (2025 test period).
Unlike RSI which lags, Fisher normalizes price to Gaussian distribution, giving cleaner reversal signals.
Long when Fisher crosses above -1.5, short when crosses below +1.5.

For 1h timeframe, we MUST limit trades to 30-60/year to avoid fee drag.
Entry requires 3+ confluence: HTF trend + regime + Fisher signal + volume.

Position sizing: 0.20 base, 0.30 strong (smaller for lower TF to reduce fee impact)
Stoploss: 2.5 * ATR trailing
Target trades: 40-70/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_hma_4h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for cleaner reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_hl = hh - ll
        if range_hl > 0:
            normalized = (hl2 - ll) / range_hl
        else:
            normalized = 0.5
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Trigger line (1-period lag)
        if i > period:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 4h HTF indicators (regime filter)
    chop_4h_14 = calculate_choppiness_index(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    chop_4h_14_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 1h HMA for local trend confirmation
    hma_1h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h TF)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50  # Ensure we don't trade immediately
    
    # Fisher crossover tracking
    prev_fisher_cross_long = False
    prev_fisher_cross_short = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(chop_4h_14_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        # Bull: price above 1d HMA(21) and HMA(21) > HMA(50)
        # Bear: price below 1d HMA(21) and HMA(21) < HMA(50)
        trend_1d_bull = close[i] > hma_1d_21_aligned[i] and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        trend_1d_bear = close[i] < hma_1d_21_aligned[i] and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        trend_1d_neutral = not trend_1d_bull and not trend_1d_bear
        
        # === 4H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_4h_14_aligned[i] > 55.0
        is_trending = chop_4h_14_aligned[i] < 45.0
        
        # === 1H LOCAL SIGNALS ===
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_21[i]
        
        # Volume filter (avoid low liquidity)
        volume_ok = vol_ratio[i] > 0.7
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_cross_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # Fisher extreme levels (stronger signals)
        fisher_extreme_long = fisher[i] < -2.0 and fisher[i] > fisher_trigger[i]
        fisher_extreme_short = fisher[i] > 2.0 and fisher[i] < fisher_trigger[i]
        
        # RSI confirmation
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG entries (require 3+ confluence)
        if volume_ok:
            # Strong long: Bull trend + trending regime + Fisher long + price above 1h HMA
            if trend_1d_bull and is_trending and fisher_cross_long and price_above_1h_hma:
                new_signal = STRONG_SIZE
            # Mean revert long: Choppy regime + Fisher extreme long + RSI oversold
            elif is_choppy and fisher_extreme_long and rsi_oversold:
                new_signal = BASE_SIZE
            # Trend pullback long: Bull trend + choppy + Fisher cross + price above 1d HMA
            elif trend_1d_bull and is_choppy and fisher_cross_long and close[i] > hma_1d_21_aligned[i]:
                new_signal = BASE_SIZE
        
        # SHORT entries (require 3+ confluence)
        if volume_ok:
            # Strong short: Bear trend + trending regime + Fisher short + price below 1h HMA
            if trend_1d_bear and is_trending and fisher_cross_short and price_below_1h_hma:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
            # Mean revert short: Choppy regime + Fisher extreme short + RSI overbought
            elif is_choppy and fisher_extreme_short and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # Trend pullback short: Bear trend + choppy + Fisher cross + price below 1d HMA
            elif trend_1d_bear and is_choppy and fisher_cross_short and close[i] < hma_1d_21_aligned[i]:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === TRADE FREQUENCY CONTROL ===
        # Only allow new position if enough bars since last trade (target 40-70 trades/year)
        bars_since_last_trade = i - last_trade_bar
        min_bars_between_trades = 24  # ~1 day on 1h = max ~365 trades/year theoretical
                                      # But with confluence filters, actual will be 40-70
        
        if bars_since_last_trade < min_bars_between_trades and new_signal != 0.0:
            # Only allow if signal is opposite to current position (exit and reverse)
            if in_position and np.sign(new_signal) == position_side:
                new_signal = 0.0
        
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
            # Long position but regime turns strongly bearish
            if position_side > 0 and trend_1d_bear and price_below_1h_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and trend_1d_bull and price_above_1h_hma:
                regime_reversal = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Long: Fisher crosses below 0 (momentum lost)
            if position_side > 0 and fisher[i] < 0.0 and fisher_trigger[i] >= 0.0:
                fisher_exit = True
            # Short: Fisher crosses above 0 (momentum lost)
            if position_side < 0 and fisher[i] > 0.0 and fisher_trigger[i] <= 0.0:
                fisher_exit = True
        
        if stoploss_triggered or regime_reversal or fisher_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        signal_changed = (signals[i-1] != new_signal) if i > 0 else True
        
        if new_signal != 0.0 and signal_changed:
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
        elif new_signal == 0.0:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
        
        # Update Fisher crossover tracking
        prev_fisher_cross_long = fisher_cross_long
        prev_fisher_cross_short = fisher_cross_short
    
    return signals