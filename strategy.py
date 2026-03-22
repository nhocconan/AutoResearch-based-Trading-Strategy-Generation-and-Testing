#!/usr/bin/env python3
"""
Experiment #007: 15m Multi-Timeframe Mean Reversion with 4h Trend Bias

Hypothesis: After analyzing 6 failed experiments, the pattern shows:
- Pure trend strategies fail in bear/range markets (2022 crash, 2025 bear)
- Lower TFs (15m) suffer from fee drag WITHOUT proper HTF filtering
- The winning strategy (#002, 30m Supertrend) used 4h HMA + RSI pullback

This 15m strategy combines:
1. 4h HMA trend bias: Ultra-stable HTF direction (changes ~2-4x/year)
2. 1h RSI pullback: Enter on retracements within HTF trend (not extremes)
3. Bollinger Band squeeze: Only enter when volatility is compressed
4. ADX(14) filter: ADX > 20 to avoid dead chop, ADX < 40 to avoid exhaustion
5. Volume confirmation: Breakout volume > 1.2 * 20bar average
6. ATR trailing stop: 2.5 * ATR(14) to protect from reversals
7. Mean reversion mode: When ADX < 20, fade BB extremes with HTF bias

Why 15m can work (unlike failed #001):
- 4h HMA bias is extremely stable (rarely changes = minimal churn)
- BB squeeze + ADX filter reduces trades to 40-80/year (Rule 10 target)
- Mean reversion mode captures range-bound periods (2025 bear market)
- Dual-mode logic: trend-follow when ADX>25, mean-revert when ADX<20
- Position sizing: 0.25-0.35 discrete (protects from 2022-style crashes)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete, ATR-scaled
Stoploss: 2.5 * ATR(14) trailing
Target trades: 50-80/year (optimal for 15m per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_bbw_adx_dual_mode_atr_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.inf))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.inf))
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100
    
    return upper.values, lower.values, sma.values, bb_width.values

def calculate_bbw_percentile(bb_width, lookback=100):
    """Calculate rolling percentile of BB Width for squeeze detection."""
    bbw_s = pd.Series(bb_width)
    bbw_percentile = bbw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if (x.max() - x.min()) > 0 else 50
    )
    return bbw_percentile.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA trend bias (ultra-stable)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h RSI for pullback entries
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger(close, 20, 2.0)
    bbw_pct = calculate_bbw_percentile(bb_width, 100)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI on 15m for mean reversion mode
    rsi_15m = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_1h_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        if np.isnan(bbw_pct[i]):
            continue
        
        # === 4H HMA TREND BIAS (Ultra-stable HTF filter) ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === ADX REGIME DETECTION ===
        is_trending = adx_14[i] > 25
        is_choppy = adx_14[i] < 20
        is_strong_trend = adx_14[i] > 35
        
        # === BOLLINGER BAND SQUEEZE ===
        # BBW percentile < 30 = compression (breakout potential)
        is_squeeze = bbw_pct[i] < 30
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # === 1H RSI PULLBACK ===
        # RSI 40-60 = healthy pullback, RSI <30 or >70 = extreme
        rsi_pullback_long = 40 <= rsi_1h_aligned[i] <= 60
        rsi_pullback_short = 40 <= rsi_1h_aligned[i] <= 60
        rsi_oversold = rsi_1h_aligned[i] < 35
        rsi_overbought = rsi_1h_aligned[i] > 65
        
        # === ATR-BASED POSITION SIZING ===
        if i > 150:
            atr_median = np.nanmedian(atr_14[150:i])
            if atr_median > 0:
                atr_ratio = atr_14[i] / atr_median
                atr_ratio = np.clip(atr_ratio, 0.5, 2.5)
                size_multiplier = 1.0 / atr_ratio
            else:
                size_multiplier = 1.0
        else:
            size_multiplier = 1.0
        
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.25, 0.35)
        
        # === DUAL-MODE ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: TREND FOLLOWING (ADX > 25, trending market)
        if is_trending and not is_strong_trend:
            # Long: bullish 4H bias + 1H RSI pullback + volume + BB squeeze
            if bull_bias and rsi_pullback_long and volume_confirmed:
                # Additional confirmation: price above BB mid
                if close[i] > bb_mid[i]:
                    new_signal = current_size
            
            # Short: bearish 4H bias + 1H RSI pullback + volume + BB squeeze
            elif bear_bias and rsi_pullback_short and volume_confirmed:
                # Additional confirmation: price below BB mid
                if close[i] < bb_mid[i]:
                    new_signal = -current_size
        
        # MODE 2: SQUEEZE BREAKOUT (highest conviction)
        elif is_squeeze and volume_confirmed:
            # Breakout long with HTF bias
            if bull_bias and close[i] > bb_upper[i] * 0.995:
                new_signal = current_size
            
            # Breakout short with HTF bias
            elif bear_bias and close[i] < bb_lower[i] * 1.005:
                new_signal = -current_size
        
        # MODE 3: MEAN REVERSION (ADX < 20, choppy/range market)
        elif is_choppy:
            # Long: oversold RSI + price at/near BB lower + bullish HTF bias
            if rsi_oversold and close[i] <= bb_lower[i] * 1.01 and bull_bias:
                new_signal = current_size * 0.7  # Smaller size for mean reversion
            
            # Short: overbought RSI + price at/near BB upper + bearish HTF bias
            elif rsi_overbought and close[i] >= bb_upper[i] * 0.99 and bear_bias:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4H bias turns bearish
            if position_side > 0 and bear_bias and adx_14[i] > 20:
                trend_reversal = True
            # Exit short if 4H bias turns bullish
            if position_side < 0 and bull_bias and adx_14[i] > 20:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (mean reversion mode) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if 1H RSI becomes overbought
            if position_side > 0 and rsi_1h_aligned[i] > 70:
                rsi_exit = True
            # Exit short if 1H RSI becomes oversold
            if position_side < 0 and rsi_1h_aligned[i] < 30:
                rsi_exit = True
        
        # Apply stoploss, trend reversal, or RSI exit
        if stoploss_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals