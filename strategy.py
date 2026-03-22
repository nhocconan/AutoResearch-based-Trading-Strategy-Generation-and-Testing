#!/usr/bin/env python3
"""
Experiment #103: 15m RSI Mean Reversion + 4h HMA Trend Filter + Volume Confirmation
Hypothesis: 15m timeframe is too noisy for pure trend following (see #091, #097 failures).
Mean reversion with HTF trend bias works better on intraday TFs.
4h HMA provides stable trend direction without excessive lag.
RSI(14) pullback entries (30-70 range, not extremes) ensure enough trades.
Volume confirmation (taker_buy_ratio > 0.55 for longs) filters false signals.
ATR(14) stoploss at 2.0*ATR protects against adverse moves.

Why this might work on 15m (learning from failed 15m experiments):
- #091 (15m Supertrend) failed: Sharpe=-0.559, too many whipsaws
- #097 (15m Supertrend + 4h HMA + 1h RSI + BB) failed: Sharpe=-1.612, too complex
- Key insight: SIMPLER entry conditions + mean reversion bias = more trades, less churn
- RSI pullback (not extremes) catches more entries than RSI<30 or RSI>70
- 4h HMA is stable enough to filter noise but fast enough for 15m entries
- Volume confirmation ensures we trade with market flow, not against it

Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.0*ATR.
Target: ≥10 trades per symbol on train, ≥3 on test, Sharpe > 0.436 (beat #100).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_meanrev_4h_hma_volume_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.zeros(n)
    mask = volume > 0
    vol_ratio[mask] = taker_buy_vol[mask] / volume[mask]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === EMA ALIGNMENT (15m) ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        # For longs in uptrend: RSI pullback to 35-50 (not oversold, just dip)
        # For shorts in downtrend: RSI rally to 50-65 (not overbought, just bounce)
        rsi_pullback_long = 35 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 65
        
        # RSI extreme mean reversion (works in range markets)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # === VOLUME CONFIRMATION ===
        # Long: taker buy ratio > 0.52 (buying pressure)
        # Short: taker buy ratio < 0.48 (selling pressure)
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.005  # within 0.5% of lower band
        near_bb_upper = close[i] >= bb_upper[i] * 0.995  # within 0.5% of upper band
        near_bb_mid = (bb_lower[i] < close[i] < bb_upper[i])
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (mean reversion in uptrend) ===
        # Path 1: 4h bullish + RSI pullback + volume confirmation (primary)
        if bull_trend_4h and rsi_pullback_long and vol_bullish:
            if ema_bullish or near_bb_lower:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: 4h bullish + RSI oversold (strong mean reversion)
        if new_signal == 0.0 and bull_trend_4h and rsi_oversold:
            if vol_bullish or ema_bullish:
                new_signal = SIZE_BASE
        
        # Path 3: 4h bullish + near BB lower + volume (BB bounce)
        if new_signal == 0.0 and bull_trend_4h and near_bb_lower:
            if vol_bullish or rsi_pullback_long:
                new_signal = SIZE_BASE
        
        # Path 4: Range market (4h neutral) + RSI oversold + BB lower (pure mean reversion)
        if new_signal == 0.0 and rsi_oversold and near_bb_lower:
            if vol_bullish:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (mean reversion in downtrend) ===
        # Path 1: 4h bearish + RSI pullback + volume confirmation (primary)
        if bear_trend_4h and rsi_pullback_short and vol_bearish:
            if ema_bearish or near_bb_upper:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: 4h bearish + RSI overbought (strong mean reversion)
        if new_signal == 0.0 and bear_trend_4h and rsi_overbought:
            if vol_bearish or ema_bearish:
                new_signal = -SIZE_BASE
        
        # Path 3: 4h bearish + near BB upper + volume (BB rejection)
        if new_signal == 0.0 and bear_trend_4h and near_bb_upper:
            if vol_bearish or rsi_pullback_short:
                new_signal = -SIZE_BASE
        
        # Path 4: Range market (4h neutral) + RSI overbought + BB upper (pure mean reversion)
        if new_signal == 0.0 and rsi_overbought and near_bb_upper:
            if vol_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR for 15m ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals